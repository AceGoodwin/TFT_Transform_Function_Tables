"""
TFT — Transform Function Table
================================
Author : Ace Goodwin  |  Date : April 2026

Run 17 — close the gap, free the servos
-----------------------------------------
Run 16 confirmed the symmetric gap control works as a restoring force
(gap declined from 7% to 5.3% over 200 epochs). But both servos were
pinned — weights at the floor, tables at the ceiling — because the gap
never closed enough to relax the multipliers.

Root cause: Dropout(0.25) is insufficient regularisation for this
architecture. With 784→512→256→10 and BatchNorm, the model has enough
capacity to memorise the training set while the tables adapt. A 7% gap
is the symptom.

Fix: Dropout 0.25 → 0.4. This directly attacks the gap.
At gap < 2%, both multipliers = 1.0 and the servos operate purely on
val_loss — which is what we want for the final phase of learning.

Also lowering the servo ceilings and floors to give the servos more
room to modulate rather than pinning at the rails:
  weight floor: 2e-5 → 1e-5  (let servo pull weights lower if needed)
  table ceiling: 1e-3 → 5e-4  (stop table LR from pinning at top)
  weight ceiling: 3e-4 → 2e-4  (consistent with lower init LR)
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
    def forward(ctx, x: torch.Tensor, table: torch.Tensor, max_temp: float) -> torch.Tensor:
        if max_temp <= 0:
            raise ValueError(f"max_temp must be > 0, got {max_temp}")
        ctx.save_for_backward(x)
        ctx.table    = table
        ctx.max_temp = max_temp
        return TFTFunction._interpolate(x, table, max_temp)

    @staticmethod
    def _interpolate(x, table, max_temp):
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
        idx_flat, frac_flat, g_flat = idx.flatten(), frac.flatten(), grad_output.flatten()
        grad_table.scatter_add_(0, idx_flat,                         (1.0 - frac_flat) * g_flat)
        grad_table.scatter_add_(0, (idx_flat + 1).clamp_max(n - 1), frac_flat          * g_flat)

        local_slope = (table[idx + 1] - table[idx]) / step
        grad_x      = torch.clamp(grad_output * local_slope, -1.0, 1.0)

        return grad_x, grad_table, None


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class TFT(nn.Module):
    """Learnable piecewise-linear activation. Place BatchNorm1d before it."""

    _INITS = {
        "leaky_relu": lambda x: torch.where(x < 0, 0.1 * x, x),
        "relu":       lambda x: torch.clamp(x, min=0),
        "linear":     lambda x: x.clone(),
        "tanh":       torch.tanh,
        "sigmoid":    torch.sigmoid,
    }

    def __init__(self, table_size: int = 256, max_temp: float = 3.0,
                 init: str = "leaky_relu") -> None:
        super().__init__()
        if init not in self._INITS:
            raise ValueError(f"init must be one of {list(self._INITS)}")
        self.table_size = table_size
        self.max_temp   = max_temp
        x_init     = torch.linspace(-max_temp, max_temp, table_size)
        self.table = nn.Parameter(self._INITS[init](x_init))
        self.history: list[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TFTFunction.apply(x, self.table, self.max_temp)

    def snapshot(self) -> None:
        self.history.append(self.table.detach().cpu().numpy().copy())

    def extra_repr(self) -> str:
        return f"table_size={self.table_size}, max_temp={self.max_temp}"


# ---------------------------------------------------------------------------
# Constraints and clipping
# ---------------------------------------------------------------------------

def _clamp_weights(model: nn.Module, headroom: float = 10.0) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            bound = headroom / (m.weight.size(1) ** 0.5)
            m.weight.data.clamp_(-bound, bound)


def _clip_group(params: list[nn.Parameter], max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = sum(g.norm(2).item() ** 2 for g in grads) ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for g in grads: g.mul_(scale)
    return total


def _update_norm(params: list[nn.Parameter],
                 prev_vals: list[torch.Tensor]) -> float:
    pairs = [(p.detach(), pv) for p, pv in zip(params, prev_vals)
             if p.grad is not None]
    if not pairs:
        return 0.0
    total = sum((cur - pv).norm(2).item() ** 2 for cur, pv in pairs)
    return (total / len(pairs)) ** 0.5


# ---------------------------------------------------------------------------
# Coupled servo controller
# ---------------------------------------------------------------------------

class CoupledLRController:
    """Cross-connected feedback-loop LR controller with gap multiplier."""

    def __init__(self, optimizer, group_idx: int,
                 lr_init: float, lr_min: float, lr_max: float,
                 ema_alpha: float = 0.2,
                 gain_up: float = 1.04, gain_down: float = 0.96,
                 rise_penalty: float = 0.85,
                 coupling_gain: float = 0.02) -> None:
        self.opt, self.idx        = optimizer, group_idx
        self.lr_min, self.lr_max  = lr_min, lr_max
        self.alpha                = ema_alpha
        self.gain_up, self.gain_down, self.rise_pen = gain_up, gain_down, rise_penalty
        self.coup_gain            = coupling_gain
        self.ema = self.prev_ema  = None
        self.cross_ema            = None
        self._set(lr_init)

    def _set(self, lr: float) -> None:
        self.opt.param_groups[self.idx]["lr"] = max(self.lr_min, min(self.lr_max, lr))

    def _get(self) -> float:
        return self.opt.param_groups[self.idx]["lr"]

    def step(self, own_metric: float, cross_gnorm: float,
             gap_multiplier: float = 1.0) -> float:
        if self.ema is None:
            self.ema = own_metric; self.prev_ema = own_metric
            self.cross_ema = cross_gnorm
            return self._get()

        self.prev_ema  = self.ema
        self.ema       = self.alpha * own_metric  + (1.0 - self.alpha) * self.ema
        delta          = (self.ema - self.prev_ema) / (self.prev_ema + 1e-8)

        prev_cross     = self.cross_ema
        self.cross_ema = self.alpha * cross_gnorm + (1.0 - self.alpha) * self.cross_ema
        cross_delta    = (self.cross_ema - prev_cross) / (prev_cross + 1e-8)

        lr = self._get()
        if   delta >  0.005: lr *= self.rise_pen
        elif delta > -0.001: lr *= self.gain_up
        else:                lr *= self.gain_down

        if cross_delta > 0.01:
            lr *= (1.0 + self.coup_gain * min(cross_delta / 0.1, 3.0))

        lr *= gap_multiplier
        self._set(lr)
        return self._get()


# ---------------------------------------------------------------------------
# Dark-mode plot helpers
# ---------------------------------------------------------------------------

_BG, _AX_BG, _FG  = "#1a1a2e", "#16213e", "#e0e0e0"
_GRID, _ZERO       = "#444444", "#666666"
_BLUE, _CORAL      = "#58b4ff", "#ff7c5c"


def _style_ax(ax: plt.Axes, title: str = "") -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=_FG, fontsize=9)


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------

def train_fashion_mnist(save_dir: str = ".") -> None:
    """
    Fashion-MNIST smoke test for TFT — run 17.

    Key change: Dropout 0.25 → 0.40
    This directly reduces the train/val gap, which relaxes both gap
    multipliers and allows the servo controllers to operate freely
    rather than pinning at their rails.

    Target outcome: gap < 3% by epoch 100, multipliers near 1.0,
    both servos free to modulate on val_loss signal alone.
    """
    save_dir   = Path(save_dir)
    CHECKPOINT = save_dir / "best_model_tft.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ------------------------------------------------------------------ data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)), test_size=0.2,
        random_state=42, stratify=full_train.targets)
    train_set    = Subset(full_train, train_idx)
    val_set      = Subset(full_train, val_idx)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    # --------------------------------------------------------------- model
    tft1 = TFT(table_size=256, max_temp=3.0, init="leaky_relu")
    tft2 = TFT(table_size=256, max_temp=3.0, init="leaky_relu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.BatchNorm1d(512),
        tft1,
        nn.Dropout(0.40),               # was 0.25 — stronger regularisation
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        tft2,
        nn.Dropout(0.40),               # was 0.25
        nn.Linear(256, 10),
    ).to(device)

    # ------------------------------------------------- parameter groups
    table_params  = [p for m in model.modules() if isinstance(m, TFT) for p in m.parameters()]
    global_params = [p for p in model.parameters() if not any(p is t for t in table_params)]
    TABLE_GROUP   = 0
    GLOBAL_GROUP  = 1

    optimizer = optim.AdamW([
        {"params": table_params,  "lr": 1e-4, "weight_decay": 0.0},
        {"params": global_params, "lr": 1e-4, "weight_decay": 3e-4},
    ])

    ctrl_table  = CoupledLRController(optimizer, TABLE_GROUP,
        lr_init=1e-4, lr_min=1e-5, lr_max=5e-4,   # ceiling lowered: 1e-3→5e-4
        ema_alpha=0.2, gain_up=1.04, gain_down=0.96,
        rise_penalty=0.85, coupling_gain=0.02)
    ctrl_global = CoupledLRController(optimizer, GLOBAL_GROUP,
        lr_init=1e-4, lr_min=1e-5, lr_max=2e-4,   # floor lowered: 2e-5→1e-5
        ema_alpha=0.2, gain_up=1.04, gain_down=0.96,
        rise_penalty=0.85, coupling_gain=0.02)

    GAP_THRESHOLD = 2.0
    GAP_SCALE     = 0.05

    CLIP_WEIGHTS    = 2.0
    CLIP_TABLES     = 0.5
    WEIGHT_HEADROOM = 10.0

    criterion      = nn.CrossEntropyLoss()
    MAX_EPOCHS     = 200
    SNAPSHOT_EVERY = 5
    PLOT_EVERY     = 10
    best_val_acc   = 0.0

    # ------------------------------------------ dark visualization
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=_BG)
    fig.suptitle("TFT Table Evolution", color=_FG, fontsize=13, fontweight="bold")
    ax_t1_shape, ax_t2_shape = axes[0][0], axes[0][1]
    ax_t1_heat,  ax_t2_heat  = axes[1][0], axes[1][1]
    for ax in axes.flat:
        _style_ax(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.pause(0.01)

    x_axis = np.linspace(-3.0, 3.0, 256)

    def update_plots(epoch: int) -> None:
        t1 = tft1.table.detach().cpu().numpy()
        t2 = tft2.table.detach().cpu().numpy()
        for ax, t, colour, label in [
            (ax_t1_shape, t1, _BLUE,  "TFT1 [BN]"),
            (ax_t2_shape, t2, _CORAL, "TFT2 [BN]"),
        ]:
            ax.clear(); _style_ax(ax, f"{label} — epoch {epoch+1}")
            ax.plot(x_axis, t, color=colour, linewidth=1.8)
            ax.axhline(0, color=_ZERO, linewidth=0.7, linestyle="--")
            ax.axvline(0, color=_ZERO, linewidth=0.7, linestyle="--")
            ax.set_xlabel("Normalised input", color=_FG)
            ax.set_ylabel("Output", color=_FG)
        for ax, tft_obj, cmap, label in [
            (ax_t1_heat, tft1, "Blues",   "TFT1"),
            (ax_t2_heat, tft2, "Oranges", "TFT2"),
        ]:
            ax.clear()
            _style_ax(ax, f"{label} — evolution ({len(tft_obj.history)} snapshots, oldest→top)")
            if len(tft_obj.history) > 1:
                mat = np.array(tft_obj.history)
                ax.imshow(mat, aspect="auto", cmap=cmap,
                          extent=[-3.0, 3.0, len(tft_obj.history), 0],
                          interpolation="nearest")
                ax.set_xlabel("Normalised input", color=_FG)
                ax.set_ylabel("Snapshot (oldest=top)", color=_FG)
        fig.canvas.draw(); fig.canvas.flush_events()

    # ---------------------------------------------------------------- loop
    prev_lr_w = prev_lr_t = None

    for epoch in tqdm(range(MAX_EPOCHS), desc="TFT training"):

        model.train()
        epoch_loss = correct = n_train = 0
        gnorm_w_sq = gnorm_t_sq = 0.0
        update_w_sq = update_t_sq = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, target)
            loss.backward()

            _clip_group(global_params, CLIP_WEIGHTS)
            gnorm_w = _clip_group(global_params, CLIP_WEIGHTS)
            gnorm_t = _clip_group(table_params,  CLIP_TABLES)
            gnorm_w_sq += gnorm_w ** 2
            gnorm_t_sq += gnorm_t ** 2

            prev_w = [p.detach().clone() for p in global_params if p.grad is not None]
            prev_t = [p.detach().clone() for p in table_params  if p.grad is not None]
            gp_act = [p for p in global_params if p.grad is not None]
            tp_act = [p for p in table_params  if p.grad is not None]

            optimizer.step()
            _clamp_weights(model, headroom=WEIGHT_HEADROOM)

            update_w_sq += _update_norm(gp_act, prev_w) ** 2
            update_t_sq += _update_norm(tp_act, prev_t) ** 2

            epoch_loss += loss.item()
            correct    += out.argmax(1).eq(target).sum().item()
            n_train    += target.size(0)

        n_b        = len(train_loader)
        gnorm_w    = (gnorm_w_sq  / n_b) ** 0.5
        gnorm_t    = (gnorm_t_sq  / n_b) ** 0.5
        update_w   = (update_w_sq / n_b) ** 0.5
        update_t   = (update_t_sq / n_b) ** 0.5
        train_loss = epoch_loss / n_b
        train_acc  = 100.0 * correct / n_train

        # -------------------------------------------------------- validate
        model.eval()
        val_loss_sum = val_correct = n_val = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                val_loss_sum += criterion(out, target).item()
                val_correct  += out.argmax(1).eq(target).sum().item()
                n_val        += target.size(0)

        val_loss = val_loss_sum / len(val_loader)
        val_acc  = 100.0 * val_correct / n_val
        gap      = train_acc - val_acc

        excess      = max(0.0, gap - GAP_THRESHOLD)
        table_mult  = 1.0 + excess * GAP_SCALE
        weight_mult = 1.0 / (1.0 + excess * GAP_SCALE)

        new_lr_t = ctrl_table.step( val_loss, gnorm_w, gap_multiplier=table_mult)
        new_lr_w = ctrl_global.step(val_loss, gnorm_t, gap_multiplier=weight_mult)

        if prev_lr_w is not None:
            frac_w   = abs(new_lr_w - prev_lr_w) / (prev_lr_w + 1e-10)
            frac_t   = abs(new_lr_t - prev_lr_t) / (prev_lr_t + 1e-10)
            locked   = abs(frac_w - frac_t) < 1e-6
            lock_str = "LOCKED" if locked else f"div {abs(frac_w-frac_t):.2e}"
        else:
            lock_str = "init"
        prev_lr_w, prev_lr_t = new_lr_w, new_lr_t

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)

        if epoch % SNAPSHOT_EVERY == 0:
            tft1.snapshot(); tft2.snapshot()

        if (epoch + 1) % PLOT_EVERY == 0:
            update_plots(epoch)

        if (epoch + 1) % 10 == 0:

            def tstats(tft: TFT) -> dict:
                t = tft.table.detach().cpu(); s = t[1:] - t[:-1]
                return {"lo": t.min().item(), "hi": t.max().item(),
                        "std": t.std().item(), "mono": (s > 0).float().mean().item(),
                        "curve": (s[1:] - s[:-1]).abs().mean().item()}

            def istats(idx: int, tft: TFT) -> dict:
                model.eval()
                with torch.no_grad():
                    d, _ = next(iter(val_loader)); x = d.to(device)
                    for i in range(idx): x = model[i](x)
                xf = x.cpu().flatten()
                w  = ((xf >= -tft.max_temp) & (xf <= tft.max_temp)).float().mean().item()
                return {"lo": xf.min().item(), "hi": xf.max().item(),
                        "std": xf.std().item(), "coverage": w}

            s1 = tstats(tft1); i1 = istats(3, tft1)
            s2 = tstats(tft2); i2 = istats(7, tft2)
            ratio = gnorm_t / (gnorm_w + 1e-8)

            print(
                f"\nEpoch {epoch+1:03d} | Loss {train_loss:.4f} | "
                f"Train {train_acc:.2f}% | Val {val_acc:.2f}% | "
                f"Gap {gap:+.2f}% | Best {best_val_acc:.2f}%\n"
                f"  LR     : weights {new_lr_w:.4e} ×{weight_mult:.3f}"
                f"   tables {new_lr_t:.4e} ×{table_mult:.3f}"
                f"   [{lock_str}]\n"
                f"  GradN  : weights {gnorm_w:.3f}   tables {gnorm_t:.4f}"
                f"   ratio {ratio:.4f}\n"
                f"  Update : weights {update_w:.2e}   tables {update_t:.2e}\n"
                f"  TFT1 [BN]  table [{s1['lo']:.3f}, {s1['hi']:.3f}]"
                f"  std {s1['std']:.3f}  mono {s1['mono']:.2f}  curve {s1['curve']:.4f}\n"
                f"  TFT2 [BN]  table [{s2['lo']:.3f}, {s2['hi']:.3f}]"
                f"  std {s2['std']:.3f}  mono {s2['mono']:.2f}  curve {s2['curve']:.4f}\n"
                f"  TFT1 input [{i1['lo']:.2f}, {i1['hi']:.2f}]"
                f"  std {i1['std']:.2f}  coverage {i1['coverage']*100:.1f}%\n"
                f"  TFT2 input [{i2['lo']:.2f}, {i2['hi']:.2f}]"
                f"  std {i2['std']:.2f}  coverage {i2['coverage']*100:.1f}%\n"
            )

    plt.ioff()
    plt.savefig(save_dir / "tft_evolution.png", dpi=120,
                bbox_inches="tight", facecolor=_BG)
    plt.show()

    if not CHECKPOINT.exists():
        print("No checkpoint found."); return

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    test_set    = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=512, num_workers=0)
    correct     = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(1).eq(target).sum().item()

    test_acc = 100.0 * correct / len(test_set)
    print(f"\n{'='*60}")
    print(f"  Final Test Accuracy (TFT) : {test_acc:.2f}%")
    print(f"  Best Val  Accuracy (TFT)  : {best_val_acc:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_fashion_mnist()