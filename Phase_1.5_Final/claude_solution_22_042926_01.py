"""
TFT — Transform Function Table  |  Phase 1.5 — Run 4
======================================================
Author : Ace Goodwin  |  Date : April 2026

What the first three runs revealed
------------------------------------
Run 1: per-group clip → weight grad explosion (3 layers too deep for 2.0 clip)
Run 2: global clip → table grads crushed to zero (530K weights dominate budget)
Run 3: separate clips → table grad norm hits the ceiling and stays pinned

Run 3 diagnosis in detail:
  TFT1 grad norm grew: 0.037 → 0.061 → 0.097 → 0.142 → 0.186 → 0.228 → CRASH
  At crash: t1 pinned at 0.5000 for 1860 of 2000 epochs.
  TFT2 pinned at 0.5000 for ~1600 epochs.
  TFT3 stayed at 0.03 — never crashed.

  The problem is not gradient *magnitude* — it's gradient *velocity*.
  When TFT1's table gradient is consistently 0.5 (at the clip ceiling),
  the table is changing shape at maximum speed every single batch.
  The weights that were trained to work with the OLD table shape now
  produce wrong intermediate representations because the activation
  function shifted under them. Network coherence collapses.

  Key insight: TFT3 (deepest layer, fewest upstream gradient paths)
  never crashes. TFT1 (first layer, most upstream gradient paths) crashes
  hardest. This confirms the issue is gradient accumulation through depth,
  not any single layer instability.

The fix: table update magnitude limiter
-----------------------------------------
Instead of (or in addition to) clipping grad norms, we limit how much
any single table entry can change per optimizer step.

After optimizer.step(), for each TFT:
  delta = table.data - table_before_step
  table.data = table_before_step + delta.clamp(-MAX_TABLE_DELTA, MAX_TABLE_DELTA)

This is a hard limit on table shape velocity regardless of grad norm,
LR, or Adam momentum. It directly prevents the "shape dragging" failure.

MAX_TABLE_DELTA = 0.02 per entry per step
  With batch_size=256 and ~188 batches/epoch:
  Max change per epoch per entry = 0.02 × 188 = 3.76
  Table range is typically [-1, 5], so max drift = 3.76 per epoch.
  This is aggressive — keeps tables from changing faster than weights.

Table LR ceiling also reduced: 5e-4 → 1e-4
  The servo was pushing tables to 5e-4 and keeping them there.
  With the delta limiter, a high LR that's immediately clamped wastes
  momentum. Lower ceiling + delta limiter = stable, controlled table learning.

Weight clip kept at 1.0 (separate from tables, working correctly).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Core autograd primitive
# ---------------------------------------------------------------------------

class TFTFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, table: torch.Tensor,
                max_temp: float) -> torch.Tensor:
        if max_temp <= 0:
            raise ValueError(f"max_temp must be > 0, got {max_temp}")
        ctx.save_for_backward(x)
        ctx.table    = table
        ctx.max_temp = max_temp
        return TFTFunction._interp(x, table, max_temp)

    @staticmethod
    def _interp(x, table, max_temp):
        n      = table.size(0)
        step   = (2.0 * max_temp) / n
        scaled = (x + max_temp) / step
        idx    = scaled.long().clamp(0, n - 2)
        frac   = (scaled - idx.float()).clamp(0.0, 1.0)
        return (1.0 - frac) * table[idx] + frac * table[idx + 1]

    @staticmethod
    def backward(ctx, grad_output):
        (x,)     = ctx.saved_tensors
        table    = ctx.table
        max_temp = ctx.max_temp
        n        = table.size(0)
        step     = (2.0 * max_temp) / n

        scaled = (x + max_temp) / step
        idx    = scaled.long().clamp(0, n - 2)
        frac   = (scaled - idx.float()).clamp(0.0, 1.0)

        grad_table = torch.zeros_like(table)
        idx_flat, frac_flat = idx.flatten(), frac.flatten()
        g_flat = grad_output.flatten()
        grad_table.scatter_add_(0, idx_flat,
                                (1.0 - frac_flat) * g_flat)
        grad_table.scatter_add_(0, (idx_flat + 1).clamp_max(n - 1),
                                frac_flat * g_flat)

        local_slope = (table[idx + 1] - table[idx]) / step
        grad_x = torch.clamp(grad_output * local_slope, -1.0, 1.0)

        return grad_x, grad_table, None


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class TFT(nn.Module):

    _INITS = {
        "leaky_relu": lambda x: torch.where(x < 0, 0.1 * x, x),
        "relu":       lambda x: torch.clamp(x, min=0),
        "linear":     lambda x: x.clone(),
        "tanh":       torch.tanh,
        "sigmoid":    torch.sigmoid,
    }

    def __init__(self, table_size: int = 256, max_temp: float = 3.0,
                 init: str = "leaky_relu", name: str = "") -> None:
        super().__init__()
        if init not in self._INITS:
            raise ValueError(f"init must be one of {list(self._INITS)}")
        self.table_size = table_size
        self.max_temp   = max_temp
        self.tft_name   = name
        x_init     = torch.linspace(-max_temp, max_temp, table_size)
        self.table = nn.Parameter(self._INITS[init](x_init))
        self.history: list[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TFTFunction.apply(x, self.table, self.max_temp)

    def snapshot(self) -> None:
        self.history.append(self.table.detach().cpu().numpy().copy())

    def extra_repr(self) -> str:
        return (f"table_size={self.table_size}, max_temp={self.max_temp}, "
                f"name={self.tft_name!r}")


# ---------------------------------------------------------------------------
# Table delta limiter — the key new constraint
# ---------------------------------------------------------------------------

def _limit_table_delta(tft: TFT, before: torch.Tensor,
                       max_delta: float) -> float:
    """
    Clamp per-entry table change since 'before'.
    Returns the mean absolute delta after clamping (for logging).
    """
    delta     = tft.table.data - before
    clamped   = delta.clamp(-max_delta, max_delta)
    tft.table.data = before + clamped
    return clamped.abs().mean().item()


# ---------------------------------------------------------------------------
# Other constraints
# ---------------------------------------------------------------------------

def _clamp_weights(model: nn.Module, headroom: float = 8.0) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            bound = headroom / (m.weight.size(1) ** 0.5)
            m.weight.data.clamp_(-bound, bound)


def _update_norm(params: list[nn.Parameter],
                 prev_vals: list[torch.Tensor]) -> float:
    pairs = [(p.detach(), pv) for p, pv in zip(params, prev_vals)
             if p.grad is not None]
    if not pairs:
        return 0.0
    return (sum((c - pv).norm(2).item() ** 2
                for c, pv in pairs) / len(pairs)) ** 0.5


def _gnorm(params: list[nn.Parameter]) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return sum(g.norm(2).item() ** 2 for g in grads) ** 0.5


# ---------------------------------------------------------------------------
# Servo controller
# ---------------------------------------------------------------------------

class CoupledLRController:
    def __init__(self, optimizer, group_idx: int,
                 lr_init: float, lr_min: float, lr_max: float,
                 ema_alpha: float = 0.2,
                 gain_up: float = 1.04, gain_down: float = 0.96,
                 rise_penalty: float = 0.85,
                 coupling_gain: float = 0.02) -> None:
        self.opt, self.idx        = optimizer, group_idx
        self.lr_min, self.lr_max  = lr_min, lr_max
        self.alpha                = ema_alpha
        self.gain_up, self.gain_down = gain_up, gain_down
        self.rise_pen             = rise_penalty
        self.coup_gain            = coupling_gain
        self.ema = self.prev_ema  = None
        self.cross_ema            = None
        self._set(lr_init)

    def _set(self, lr: float) -> None:
        self.opt.param_groups[self.idx]["lr"] = (
            max(self.lr_min, min(self.lr_max, lr)))

    def _get(self) -> float:
        return self.opt.param_groups[self.idx]["lr"]

    def step(self, own_metric: float, cross_gnorm: float,
             gap_multiplier: float = 1.0) -> float:
        if self.ema is None:
            self.ema = own_metric; self.prev_ema = own_metric
            self.cross_ema = cross_gnorm
            return self._get()

        self.prev_ema  = self.ema
        self.ema       = self.alpha * own_metric + (1 - self.alpha) * self.ema
        delta          = (self.ema - self.prev_ema) / (self.prev_ema + 1e-8)

        prev_cross     = self.cross_ema
        self.cross_ema = (self.alpha * cross_gnorm
                          + (1 - self.alpha) * self.cross_ema)
        cross_delta    = (self.cross_ema - prev_cross) / (prev_cross + 1e-8)

        lr = self._get()
        if   delta >  0.005: lr *= self.rise_pen
        elif delta > -0.001: lr *= self.gain_up
        else:                lr *= self.gain_down

        if cross_delta > 0.01:
            lr *= (1 + self.coup_gain * min(cross_delta / 0.1, 3.0))

        lr *= gap_multiplier
        self._set(lr)
        return self._get()


# ---------------------------------------------------------------------------
# Dark-mode visualisation
# ---------------------------------------------------------------------------

_BG, _AX_BG, _FG = "#1a1a2e", "#16213e", "#e0e0e0"
_GRID, _ZERO     = "#444444", "#666666"
_COLOURS         = ["#58b4ff", "#ff7c5c", "#7dff9e"]
_CMAPS           = ["Blues", "Oranges", "Greens"]


def _style_ax(ax, title=""):
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG); ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)
    if title: ax.set_title(title, color=_FG, fontsize=8)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fashion_mnist(save_dir: str = ".") -> None:
    save_dir   = Path(save_dir)
    CHECKPOINT = save_dir / "best_model_tft.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)), test_size=0.2,
        random_state=42, stratify=full_train.targets)
    train_loader = DataLoader(Subset(full_train, train_idx),
                              batch_size=256, shuffle=True, num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(Subset(full_train, val_idx),
                              batch_size=512, shuffle=False, num_workers=0,
                              pin_memory=True)

    # ------------------------------------------------------------ model
    tft1 = TFT(table_size=256, max_temp=3.5, init="leaky_relu", name="TFT1")
    tft2 = TFT(table_size=256, max_temp=3.0, init="tanh",       name="TFT2")
    tft3 = TFT(table_size=256, max_temp=3.0, init="relu",       name="TFT3")
    tfts = [tft1, tft2, tft3]

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.BatchNorm1d(512), tft1, nn.Dropout(0.40),
        nn.Linear(512, 256), nn.BatchNorm1d(256), tft2, nn.Dropout(0.40),
        nn.Linear(256, 128), nn.BatchNorm1d(128), tft3, nn.Dropout(0.30),
        nn.Linear(128, 10),
    ).to(device)

    tft1_params   = list(tft1.parameters())
    tft2_params   = list(tft2.parameters())
    tft3_params   = list(tft3.parameters())
    all_tft_ids   = set(id(p) for p in tft1_params+tft2_params+tft3_params)
    weight_params = [p for p in model.parameters() if id(p) not in all_tft_ids]
    tft_param_groups = [tft1_params, tft2_params, tft3_params]

    T1, T2, T3, WG = 0, 1, 2, 3

    optimizer = optim.AdamW([
        {"params": tft1_params,   "lr": 1e-4, "weight_decay": 0.0},
        {"params": tft2_params,   "lr": 1e-4, "weight_decay": 0.0},
        {"params": tft3_params,   "lr": 1e-4, "weight_decay": 0.0},
        {"params": weight_params, "lr": 1e-4, "weight_decay": 3e-4},
    ])

    def _mkctrl(grp, lr_max):
        return CoupledLRController(optimizer, grp,
            lr_init=1e-4, lr_min=1e-5, lr_max=lr_max,
            ema_alpha=0.2, gain_up=1.04, gain_down=0.96,
            rise_penalty=0.85, coupling_gain=0.02)

    # Table LR ceiling lowered to 1e-4 (from 5e-4 in run 3).
    # Combined with delta limiter, this prevents runaway table velocity.
    ctrl_t1 = _mkctrl(T1, lr_max=1e-4)
    ctrl_t2 = _mkctrl(T2, lr_max=1e-4)
    ctrl_t3 = _mkctrl(T3, lr_max=1e-4)
    ctrl_w  = CoupledLRController(optimizer, WG,
        lr_init=1e-4, lr_min=1e-5, lr_max=2e-4,
        ema_alpha=0.2, gain_up=1.04, gain_down=0.96,
        rise_penalty=0.85, coupling_gain=0.02)

    GAP_THRESHOLD  = 2.0
    GAP_SCALE      = 0.05 
    WEIGHT_CLIP    = 1.0
    TABLE_CLIP     = 0.5
    WEIGHT_HEADROOM = 8.0
    # Table update delta limiter — max change per entry per step
    MAX_TABLE_DELTA = 0.02

    criterion      = nn.CrossEntropyLoss()
    MAX_EPOCHS     = 2000
    SNAPSHOT_EVERY = 5
    PLOT_EVERY     = 10
    best_val_acc   = 0.0

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), facecolor=_BG)
    fig.suptitle("TFT Phase 1.5 — Per-Layer Table Evolution",
                 color=_FG, fontsize=13, fontweight="bold")
    for ax in axes.flat: _style_ax(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.pause(0.01)

    x_axes = [np.linspace(-t.max_temp, t.max_temp, 256) for t in tfts]
    inits  = ["leaky_relu", "tanh", "relu"]

    def update_plots(epoch):
        for i, (tft, ax, col, xa) in enumerate(
                zip(tfts, axes[0], _COLOURS, x_axes)):
            t = tft.table.detach().cpu().numpy()
            ax.clear(); _style_ax(ax, f"TFT{i+1} [{inits[i]}] — ep {epoch+1}")
            ax.plot(xa, t, linewidth=1.8, color=col)
            ax.axhline(0, color=_ZERO, lw=0.7, ls="--")
            ax.axvline(0, color=_ZERO, lw=0.7, ls="--")
            ax.set_xlabel("Normalised input", color=_FG)
            ax.set_ylabel("Output", color=_FG)
        for i, (tft, ax, cmap) in enumerate(
                zip(tfts, axes[1], _CMAPS)):
            ax.clear()
            _style_ax(ax, f"TFT{i+1} ({len(tft.history)} snaps)")
            if len(tft.history) > 1:
                mat = np.array(tft.history)
                ax.imshow(mat, aspect="auto", cmap=cmap,
                          extent=[-tft.max_temp, tft.max_temp,
                                  len(tft.history), 0],
                          interpolation="nearest")
                ax.set_xlabel("Normalised input", color=_FG)
                ax.set_ylabel("Snapshot (oldest=top)", color=_FG)
        fig.canvas.draw(); fig.canvas.flush_events()

    prev_lrs = [None] * 4

    for epoch in tqdm(range(MAX_EPOCHS), desc="TFT-1.5 training"):

        model.train()
        epoch_loss = correct = n_train = 0
        gnorm_w_sq = 0.0; gnorm_t_sq = [0.0]*3
        update_w_sq = 0.0; update_t_sq = [0.0]*3
        delta_means = [0.0]*3   # mean per-entry delta per batch

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, target)
            loss.backward()

            # Separate clip budgets
            for tp in tft_param_groups:
                nn.utils.clip_grad_norm_(tp, TABLE_CLIP)
            nn.utils.clip_grad_norm_(weight_params, WEIGHT_CLIP)

            gn_w  = _gnorm(weight_params)
            gn_ts = [_gnorm(tp) for tp in tft_param_groups]
            gnorm_w_sq += gn_w ** 2
            for i in range(3): gnorm_t_sq[i] += gn_ts[i] ** 2

            prev_t = [[p.detach().clone() for p in tp if p.grad is not None]
                      for tp in tft_param_groups]
            prev_w  = [p.detach().clone()
                       for p in weight_params if p.grad is not None]

            # Save table values before step for delta limiter
            tbl_before = [tft.table.data.clone() for tft in tfts]

            optimizer.step()

            # Apply table delta limiter
            for i, tft in enumerate(tfts):
                dm = _limit_table_delta(tft, tbl_before[i], MAX_TABLE_DELTA)
                delta_means[i] += dm

            _clamp_weights(model, headroom=WEIGHT_HEADROOM)

            act_t = [[p for p in tp if p.grad is not None]
                     for tp in tft_param_groups]
            act_w  = [p for p in weight_params if p.grad is not None]
            for i in range(3):
                update_t_sq[i] += _update_norm(act_t[i], prev_t[i]) ** 2
            update_w_sq += _update_norm(act_w, prev_w) ** 2

            epoch_loss += loss.item(); correct += out.argmax(1).eq(target).sum().item()
            n_train    += target.size(0)

        n_b       = len(train_loader)
        gnorm_w   = (gnorm_w_sq  / n_b) ** 0.5
        gnorm_ts  = [(gnorm_t_sq[i]  / n_b) ** 0.5 for i in range(3)]
        update_w  = (update_w_sq / n_b) ** 0.5
        update_ts = [(update_t_sq[i] / n_b) ** 0.5 for i in range(3)]
        dm_avg    = [delta_means[i] / n_b for i in range(3)]
        train_loss = epoch_loss / n_b
        train_acc  = 100.0 * correct / n_train

        model.eval()
        val_ls = val_c = n_val = 0
        with torch.no_grad():
            for d, t in val_loader:
                d, t = d.to(device), t.to(device)
                o = model(d)
                val_ls += criterion(o, t).item()
                val_c  += o.argmax(1).eq(t).sum().item()
                n_val  += t.size(0)

        val_loss = val_ls / len(val_loader)
        val_acc  = 100.0 * val_c / n_val
        gap      = train_acc - val_acc

        excess      = max(0.0, gap - GAP_THRESHOLD)
        table_mult  = 1.0 + excess * GAP_SCALE
        weight_mult = 1.0 / (1.0 + excess * GAP_SCALE)

        avg_gt = sum(gnorm_ts) / 3.0
        lr_t1 = ctrl_t1.step(val_loss, gnorm_w, table_mult)
        lr_t2 = ctrl_t2.step(val_loss, gnorm_w, table_mult)
        lr_t3 = ctrl_t3.step(val_loss, gnorm_w, table_mult)
        lr_w  = ctrl_w.step( val_loss, avg_gt,  weight_mult)
        nlrs  = [lr_t1, lr_t2, lr_t3, lr_w]

        if prev_lrs[0] is not None:
            fracs    = [abs(nlrs[i]-prev_lrs[i])/(prev_lrs[i]+1e-10) for i in range(4)]
            lock_str = ("LOCKED" if max(fracs)-min(fracs) < 1e-6
                        else f"div {max(fracs)-min(fracs):.2e}")
        else:
            lock_str = "init"
        prev_lrs = nlrs[:]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)

        if epoch % SNAPSHOT_EVERY == 0:
            for tft in tfts: tft.snapshot()
        if (epoch + 1) % PLOT_EVERY == 0:
            update_plots(epoch)

        if (epoch + 1) % 10 == 0:
            def ts(tft):
                t = tft.table.detach().cpu(); s = t[1:]-t[:-1]
                return {"lo": t.min().item(), "hi": t.max().item(),
                        "std": t.std().item(), "mono": (s>0).float().mean().item(),
                        "curve": (s[1:]-s[:-1]).abs().mean().item()}

            def ist(lidx, tft):
                model.eval()
                with torch.no_grad():
                    d, _ = next(iter(val_loader)); x = d.to(device)
                    for i in range(lidx): x = model[i](x)
                xf = x.cpu().flatten()
                cov = ((xf>=-tft.max_temp)&(xf<=tft.max_temp)).float().mean().item()
                return {"lo":xf.min().item(),"hi":xf.max().item(),
                        "std":xf.std().item(),"cov":cov}

            s1=ts(tft1); s2=ts(tft2); s3=ts(tft3)
            i1=ist(3,tft1); i2=ist(7,tft2); i3=ist(11,tft3)
            ratios = [gnorm_ts[i]/(gnorm_w+1e-8) for i in range(3)]

            print(
                f"\nEpoch {epoch+1:03d} | Loss {train_loss:.4f} | "
                f"Train {train_acc:.2f}% | Val {val_acc:.2f}% | "
                f"Gap {gap:+.2f}% | Best {best_val_acc:.2f}%\n"
                f"  LR  : w {lr_w:.3e} ×{weight_mult:.3f}"
                f"  t1 {lr_t1:.3e}  t2 {lr_t2:.3e}"
                f"  t3 {lr_t3:.3e} ×{table_mult:.3f}  [{lock_str}]\n"
                f"  GradN: w {gnorm_w:.4f}"
                f"  t1 {gnorm_ts[0]:.4f}  t2 {gnorm_ts[1]:.4f}"
                f"  t3 {gnorm_ts[2]:.4f}"
                f"  ratios {ratios[0]:.3f}/{ratios[1]:.3f}/{ratios[2]:.3f}\n"
                f"  Update: w {update_w:.2e}"
                f"  t1 {update_ts[0]:.2e}  t2 {update_ts[1]:.2e}"
                f"  t3 {update_ts[2]:.2e}\n"
                f"  Delta(mean): t1 {dm_avg[0]:.4f}  t2 {dm_avg[1]:.4f}"
                f"  t3 {dm_avg[2]:.4f}  [max {MAX_TABLE_DELTA}]\n"
                f"  TFT1[lr] [{s1['lo']:.3f},{s1['hi']:.3f}]"
                f"  std {s1['std']:.3f}  mono {s1['mono']:.2f}"
                f"  curve {s1['curve']:.4f}\n"
                f"  TFT2[th] [{s2['lo']:.3f},{s2['hi']:.3f}]"
                f"  std {s2['std']:.3f}  mono {s2['mono']:.2f}"
                f"  curve {s2['curve']:.4f}\n"
                f"  TFT3[rl] [{s3['lo']:.3f},{s3['hi']:.3f}]"
                f"  std {s3['std']:.3f}  mono {s3['mono']:.2f}"
                f"  curve {s3['curve']:.4f}\n"
                f"  TFT1 input [{i1['lo']:.2f},{i1['hi']:.2f}]"
                f"  std {i1['std']:.2f}  cov {i1['cov']*100:.1f}%\n"
                f"  TFT2 input [{i2['lo']:.2f},{i2['hi']:.2f}]"
                f"  std {i2['std']:.2f}  cov {i2['cov']*100:.1f}%\n"
                f"  TFT3 input [{i3['lo']:.2f},{i3['hi']:.2f}]"
                f"  std {i3['std']:.2f}  cov {i3['cov']*100:.1f}%\n"
            )

    plt.ioff()
    plt.savefig(save_dir / "tft_p15_evolution.png", dpi=120,
                bbox_inches="tight", facecolor=_BG)
    plt.show()

    if not CHECKPOINT.exists():
        print("No checkpoint found."); return

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    test_loader = DataLoader(
        datasets.FashionMNIST(root="./data", train=False, download=True,
                              transform=transform),
        batch_size=512, num_workers=0)
    correct = 0
    with torch.no_grad():
        for d, t in test_loader:
            d, t = d.to(device), t.to(device)
            correct += model(d).argmax(1).eq(t).sum().item()

    test_acc = 100.0 * correct / 10000
    print(f"\n{'='*60}")
    print(f"  Final Test Accuracy (TFT Phase 1.5) : {test_acc:.2f}%")
    print(f"  Best Val  Accuracy (TFT Phase 1.5)  : {best_val_acc:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_fashion_mnist()