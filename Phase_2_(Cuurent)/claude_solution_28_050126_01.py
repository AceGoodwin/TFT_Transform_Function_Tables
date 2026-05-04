"""
TFT — Transform Function Table  |  Phase 2 — Run 5
====================================================
Author : Ace Goodwin  |  Date : April 2026

The diversity sweet spot is confirmed
--------------------------------------
Run 4 data:
  Epoch 110: diversity mean 0.060, val 90.97%  ← peak
  Epoch 160: diversity mean 0.100, val 90.99%  ← marginal improvement
  Epoch 400: diversity mean 0.190, val ~90.8%  ← declining

Best val occurred at diversity 0.06-0.10. After that, nodes over-specialise
and the network begins fitting training-specific table shapes.

Run 5 strategy: stop when it's good
--------------------------------------
Simple patience-based early stopping on val accuracy.
PATIENCE = 50 epochs without improvement.

This is the right stopping criterion because:
  - Val accuracy is the direct optimisation target
  - Diversity is a diagnostic, not a stopping signal
  - At Dropout 0.50, the val signal is clean enough to trust

Label smoothing 0.1 kept (working well — gap held at 5-6%).
Everything else identical to Run 4.

Goal: lock in the 90.99%+ result cleanly, get the best test score.
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
# BatchedTFT
# ---------------------------------------------------------------------------

class BatchedTFTFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tables, max_temp):
        ctx.save_for_backward(x)
        ctx.tables = tables; ctx.max_temp = max_temp
        n, ts = tables.shape; step = (2.0 * max_temp) / ts
        scaled = (x + max_temp) / step
        idx = scaled.long().clamp(0, ts - 2)
        frac = (scaled - idx.float()).clamp(0.0, 1.0)
        nr = torch.arange(n, device=x.device)
        lo = tables[nr, idx]; hi = tables[nr, (idx+1).clamp_max(ts-1)]
        return (1.0 - frac) * lo + frac * hi

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        tables = ctx.tables; max_temp = ctx.max_temp
        n, ts = tables.shape; step = (2.0 * max_temp) / ts
        scaled = (x + max_temp) / step
        idx = scaled.long().clamp(0, ts - 2)
        frac = (scaled - idx.float()).clamp(0.0, 1.0)
        grad_tables = torch.zeros_like(tables)
        b = x.size(0)
        nr = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        fl = (nr * ts + idx).flatten()
        fh = (nr * ts + (idx+1).clamp_max(ts-1)).flatten()
        gf = grad_output.flatten(); ff = frac.flatten()
        grad_tables.flatten().scatter_add_(0, fl, (1.0 - ff) * gf)
        grad_tables.flatten().scatter_add_(0, fh, ff * gf)
        nr2 = torch.arange(n, device=x.device)
        slope = (tables[nr2, (idx+1).clamp_max(ts-1)] - tables[nr2, idx]) / step
        grad_x = torch.clamp(grad_output * slope, -1.0, 1.0)
        return grad_x, grad_tables, None


class BatchedTFT(nn.Module):

    _INITS = {
        "leaky_relu": lambda x: torch.where(x < 0, 0.1 * x, x),
        "relu":       lambda x: torch.clamp(x, min=0),
        "linear":     lambda x: x.clone(),
        "tanh":       torch.tanh,
        "sigmoid":    torch.sigmoid,
    }

    def __init__(self, n_nodes, table_size=64, max_temp=3.0,
                 init="leaky_relu", name=""):
        super().__init__()
        if init not in self._INITS: raise ValueError(f"Unknown init: {init}")
        self.n_nodes = n_nodes; self.table_size = table_size
        self.max_temp = max_temp; self.tft_name = name
        xi = torch.linspace(-max_temp, max_temp, table_size)
        self.tables = nn.Parameter(
            self._INITS[init](xi).unsqueeze(0).expand(n_nodes, -1).clone())
        self.history_mean: list[np.ndarray] = []
        self.history_std:  list[np.ndarray] = []

    def forward(self, x):
        return BatchedTFTFunction.apply(x, self.tables, self.max_temp)

    def snapshot(self):
        t = self.tables.detach().cpu()
        self.history_mean.append(t.mean(0).numpy().copy())
        self.history_std.append( t.std(0).numpy().copy())

    def diversity(self):
        return self.tables.detach().std(0).mean().item()

    def extra_repr(self):
        return (f"n_nodes={self.n_nodes}, table_size={self.table_size}, "
                f"max_temp={self.max_temp}, name={self.tft_name!r}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp_weights(model, headroom=8.0):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            b = headroom / (m.weight.size(1) ** 0.5)
            m.weight.data.clamp_(-b, b)


def _gnorm(params):
    g = [p.grad for p in params if p.grad is not None]
    return sum(x.norm(2).item()**2 for x in g)**0.5 if g else 0.0


def _update_norm(params, prev):
    pairs = [(p.detach(), pv) for p, pv in zip(params, prev) if p.grad is not None]
    if not pairs: return 0.0
    return (sum((c-pv).norm(2).item()**2 for c,pv in pairs)/len(pairs))**0.5


# ---------------------------------------------------------------------------
# Servo
# ---------------------------------------------------------------------------

class CoupledLRController:
    def __init__(self, opt, grp, lr_init, lr_min, lr_max,
                 a=0.2, gu=1.04, gd=0.96, rp=0.85, cg=0.02):
        self.opt=opt; self.idx=grp
        self.lr_min=lr_min; self.lr_max=lr_max
        self.a=a; self.gu=gu; self.gd=gd; self.rp=rp; self.cg=cg
        self.ema=self.pema=None; self.cema=None; self._set(lr_init)

    def _set(self, lr):
        self.opt.param_groups[self.idx]["lr"] = max(self.lr_min, min(self.lr_max, lr))

    def _get(self): return self.opt.param_groups[self.idx]["lr"]

    def step(self, own, cross, gm=1.0):
        if self.ema is None:
            self.ema=own; self.pema=own; self.cema=cross; return self._get()
        self.pema=self.ema; self.ema=self.a*own+(1-self.a)*self.ema
        d=(self.ema-self.pema)/(self.pema+1e-8)
        pc=self.cema; self.cema=self.a*cross+(1-self.a)*self.cema
        cd=(self.cema-pc)/(pc+1e-8)
        lr=self._get()
        if   d >  0.005: lr*=self.rp
        elif d > -0.001: lr*=self.gu
        else:            lr*=self.gd
        if cd>0.01: lr*=(1+self.cg*min(cd/0.1,3.0))
        lr*=gm; self._set(lr); return self._get()


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

_BG,_AX_BG,_FG="#1a1a2e","#16213e","#e0e0e0"
_GRID,_ZERO="#444444","#666666"
_COLOURS=["#58b4ff","#ff7c5c","#7dff9e"]
_CMAPS=["Blues","Oranges","Greens"]

def _sax(ax, title=""):
    ax.set_facecolor(_AX_BG); ax.tick_params(colors=_FG,labelsize=8)
    ax.xaxis.label.set_color(_FG); ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(True,color=_GRID,lw=0.5,alpha=0.7)
    if title: ax.set_title(title,color=_FG,fontsize=8)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fashion_mnist(save_dir="."):
    save_dir = Path(save_dir)
    CHECKPOINT = save_dir / "best_model_tft_p2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(root="./data", train=True,
                                       download=True, transform=transform)
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)), test_size=0.2,
        random_state=42, stratify=full_train.targets)
    train_loader = DataLoader(Subset(full_train, train_idx),
                              batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(Subset(full_train, val_idx),
                              batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    btft1 = BatchedTFT(512, 64, 3.5, "leaky_relu", "BTFT1")
    btft2 = BatchedTFT(256, 64, 3.0, "leaky_relu", "BTFT2")
    btft3 = BatchedTFT(128, 64, 3.0, "leaky_relu", "BTFT3")
    btfts = [btft1, btft2, btft3]

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,512), nn.BatchNorm1d(512), btft1, nn.Dropout(0.50),
        nn.Linear(512,256), nn.BatchNorm1d(256), btft2, nn.Dropout(0.50),
        nn.Linear(256,128), nn.BatchNorm1d(128), btft3, nn.Dropout(0.40),
        nn.Linear(128,10),
    ).to(device)

    t1p=list(btft1.parameters()); t2p=list(btft2.parameters()); t3p=list(btft3.parameters())
    atids=set(id(p) for p in t1p+t2p+t3p)
    wp=[p for p in model.parameters() if id(p) not in atids]
    tpg=[t1p,t2p,t3p]

    optimizer = optim.AdamW([
        {"params":t1p,"lr":1e-4,"weight_decay":5e-4},
        {"params":t2p,"lr":1e-4,"weight_decay":5e-4},
        {"params":t3p,"lr":1e-4,"weight_decay":5e-4},
        {"params":wp, "lr":1e-4,"weight_decay":5e-4},
    ])

    def _mkc(g): return CoupledLRController(optimizer,g,1e-4,1e-5,3e-4)
    ct1=_mkc(0); ct2=_mkc(1); ct3=_mkc(2)
    cw=CoupledLRController(optimizer,3,1e-4,1e-5,2e-4)

    GAP_TH=2.0; GAP_SC=0.05; WCLIP=1.0; TCLIP=2.0; WH=8.0

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    MAX_EPOCHS = 500
    PATIENCE   = 50   # stop after 50 epochs without val improvement
    SNAP=5; PLOT=10
    best=0.0; no_improve=0

    plt.style.use("dark_background")
    fig,axes=plt.subplots(2,3,figsize=(18,8),facecolor=_BG)
    fig.suptitle("TFT Phase 2 R5 — Patience Early Stop",
                 color=_FG, fontsize=13, fontweight="bold")
    for ax in axes.flat: _sax(ax)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.pause(0.01)
    labels=["BTFT1 (512)","BTFT2 (256)","BTFT3 (128)"]
    xax=[np.linspace(-t.max_temp,t.max_temp,t.table_size) for t in btfts]

    def uplot(ep):
        for i,(tft,ax,col,xa,lb) in enumerate(zip(btfts,axes[0],_COLOURS,xax,labels)):
            t=tft.tables.detach().cpu(); mu=t.mean(0).numpy(); sg=t.std(0).numpy()
            ax.clear(); _sax(ax,f"{lb} ep{ep+1}")
            ax.fill_between(xa,mu-sg,mu+sg,alpha=0.25,color=col)
            ax.plot(xa,mu,lw=1.8,color=col)
            ax.axhline(0,color=_ZERO,lw=0.7,ls="--")
            ax.axvline(0,color=_ZERO,lw=0.7,ls="--")
            ax.set_xlabel("Norm input",color=_FG); ax.set_ylabel("Output",color=_FG)
        for i,(tft,ax,cm,lb) in enumerate(zip(btfts,axes[1],_CMAPS,labels)):
            ax.clear(); _sax(ax,f"{lb} ({len(tft.history_mean)} snaps)")
            if len(tft.history_mean)>1:
                mat=np.array(tft.history_mean)
                ax.imshow(mat,aspect="auto",cmap=cm,
                          extent=[-tft.max_temp,tft.max_temp,len(tft.history_mean),0],
                          interpolation="nearest")
                ax.set_xlabel("Norm input",color=_FG); ax.set_ylabel("Snapshot",color=_FG)
        fig.canvas.draw(); fig.canvas.flush_events()

    plr=[None]*4

    for epoch in tqdm(range(MAX_EPOCHS), desc="TFT-P2-R5"):
        model.train()
        el=c=nt=0; gws=0.0; gts=[0.0]*3; uws=0.0; uts=[0.0]*3
        for d,t in train_loader:
            d,t=d.to(device),t.to(device)
            optimizer.zero_grad()
            o=model(d); l=criterion(o,t); l.backward()
            for tp in tpg: nn.utils.clip_grad_norm_(tp,TCLIP)
            nn.utils.clip_grad_norm_(wp,WCLIP)
            gw=_gnorm(wp); gtsb=[_gnorm(tp) for tp in tpg]
            gws+=gw**2
            for i in range(3): gts[i]+=gtsb[i]**2
            pt=[[p.detach().clone() for p in tp if p.grad is not None] for tp in tpg]
            pw=[p.detach().clone() for p in wp if p.grad is not None]
            optimizer.step(); _clamp_weights(model,WH)
            at=[[p for p in tp if p.grad is not None] for tp in tpg]
            aw=[p for p in wp if p.grad is not None]
            for i in range(3): uts[i]+=_update_norm(at[i],pt[i])**2
            uws+=_update_norm(aw,pw)**2
            el+=l.item(); c+=o.argmax(1).eq(t).sum().item(); nt+=t.size(0)

        nb=len(train_loader)
        gw=(gws/nb)**0.5; gtse=[(gts[i]/nb)**0.5 for i in range(3)]
        uw=(uws/nb)**0.5;  utse=[(uts[i]/nb)**0.5 for i in range(3)]
        trl=el/nb; tra=100.0*c/nt

        model.eval(); vl=vc=nv=0
        with torch.no_grad():
            for d,t in val_loader:
                d,t=d.to(device),t.to(device); o=model(d)
                vl+=criterion(o,t).item()
                vc+=o.argmax(1).eq(t).sum().item(); nv+=t.size(0)
        vll=vl/len(val_loader); va=100.0*vc/nv; gap=tra-va

        # Early stopping
        if va > best:
            best=va; no_improve=0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            no_improve += 1

        ex=max(0.0,gap-GAP_TH); tm=1.0+ex*GAP_SC; wm=1.0/(1.0+ex*GAP_SC)
        ag=sum(gtse)/3.0
        lt1=ct1.step(vll,gw,tm); lt2=ct2.step(vll,gw,tm)
        lt3=ct3.step(vll,gw,tm); lw=cw.step(vll,ag,wm)
        nl=[lt1,lt2,lt3,lw]
        if plr[0] is not None:
            fr=[abs(nl[i]-plr[i])/(plr[i]+1e-10) for i in range(4)]
            ls="LOCKED" if max(fr)-min(fr)<1e-6 else f"div {max(fr)-min(fr):.2e}"
        else: ls="init"
        plr=nl[:]

        if epoch%SNAP==0:
            for tft in btfts: tft.snapshot()
        if (epoch+1)%PLOT==0: uplot(epoch)

        if (epoch+1)%10==0:
            divs=[t.diversity() for t in btfts]
            mean_div=sum(divs)/3.0
            def ist(li,tft):
                model.eval()
                with torch.no_grad():
                    d,_=next(iter(val_loader)); x=d.to(device)
                    for i in range(li): x=model[i](x)
                xf=x.cpu().flatten()
                cv=((xf>=-tft.max_temp)&(xf<=tft.max_temp)).float().mean().item()
                return {"std":xf.std().item(),"cov":cv}
            i1=ist(3,btft1); i2=ist(7,btft2); i3=ist(11,btft3)
            ra=[gtse[i]/(gw+1e-8) for i in range(3)]
            print(
                f"\nEpoch {epoch+1:03d} | Loss {trl:.4f} | "
                f"Train {tra:.2f}% | Val {va:.2f}% | Gap {gap:+.2f}% | "
                f"Best {best:.2f}% [no_improve {no_improve}/{PATIENCE}]\n"
                f"  LR : w {lw:.3e}×{wm:.3f}  t {lt1:.3e}/{lt2:.3e}/{lt3:.3e}×{tm:.3f}  [{ls}]\n"
                f"  GradN: w {gw:.4f}  t {gtse[0]:.4f}/{gtse[1]:.4f}/{gtse[2]:.4f}"
                f"  ratios {ra[0]:.3f}/{ra[1]:.3f}/{ra[2]:.3f}\n"
                f"  Diversity: {divs[0]:.4f}/{divs[1]:.4f}/{divs[2]:.4f}"
                f"  mean {mean_div:.4f}\n"
                f"  cov: BTFT1 {i1['cov']*100:.1f}% std{i1['std']:.2f}"
                f"  BTFT2 {i2['cov']*100:.1f}% std{i2['std']:.2f}"
                f"  BTFT3 {i3['cov']*100:.1f}% std{i3['std']:.2f}\n"
            )

        if no_improve >= PATIENCE:
            print(f"\n*** Early stop epoch {epoch+1}: "
                  f"{no_improve} epochs without improvement. Best {best:.2f}% ***")
            break

    # Final plot at stopping point
    uplot(epoch)
    plt.ioff()
    plt.savefig(save_dir/"tft_p2_r5_evolution.png",dpi=120,bbox_inches="tight",facecolor=_BG)
    plt.show()

    if not CHECKPOINT.exists(): print("No checkpoint."); return
    model.load_state_dict(torch.load(CHECKPOINT,map_location=device)); model.eval()
    tl=DataLoader(datasets.FashionMNIST(root="./data",train=False,
                  download=True,transform=transform),batch_size=512,num_workers=0)
    correct=0
    with torch.no_grad():
        for d,t in tl:
            d,t=d.to(device),t.to(device); correct+=model(d).argmax(1).eq(t).sum().item()
    print(f"\n{'='*60}")
    print(f"  Final Test Accuracy (TFT Phase 2 R5) : {100.0*correct/10000:.2f}%")
    print(f"  Best Val  Accuracy (TFT Phase 2 R5)  : {best:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_fashion_mnist()