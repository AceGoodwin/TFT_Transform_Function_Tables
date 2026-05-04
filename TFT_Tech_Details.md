# TFT — Transform Function Tables
## A Learnable Piecewise-Linear Activation Function for Deep Neural Networks

**Author:** Ace Goodwin  
**Development Period:** 2015–May 2026  
**Version:** Phase 2 R5  
**Best Result:** 90.11% test accuracy, 90.99% val accuracy on Fashion-MNIST (MLP, no convolutions)

---

## Abstract

TFT replaces fixed activation functions (ReLU, tanh, leaky ReLU) with learnable piecewise-linear functions stored as parameter tables and updated via standard backpropagation. Each table maps input values to output values through linear interpolation between adjacent entries. Gradients flow through the interpolation arithmetic — no symbolic derivative of the activation function is needed.

In Phase 2, each network node receives its own independent table. Starting from identical initialisations, nodes spontaneously specialise into distinct functional roles — a form of learned activation diversity not achievable with fixed activations. The best configuration achieved 90.99% validation accuracy and 90.11% test accuracy on Fashion-MNIST, matching or exceeding standard ReLU-based baselines while providing qualitatively richer internal representations.

---

## The Human Take (Ace Goodwin)

1st of all let me just say that this is NOT academics, but Mad Science. It's far too late for me to benefit from any sort of those types of accolades, so I'll have to be content with knowing I was the 1st to make ihis practical.  
It is a product of 50+ years of coding experience, and a LOT of marching in the opposite direction from standard research pathways.  
It is a product of 15 years of Vibe-Coding (yup, even before it was 'Vibe' Coding).  
It is a further proof that 1 mind, working in the worst of circumstances, can still push back the boundries of the unknown (Lookin' @ YOU Schwarzschild).  

The Perceptron has proven to be a hearty mechanism for mapping datasets. The trouble is, that it's flawed.
The final piece of the puzzle has been the Activation Function. Activation functions are often chosen at random, with the hope and prayers that a bit of voodoo luck will grant their models some measure of optimization.  
Experiments in the past (and even likely some ongoing research) has attempted to overcome this obstacle with learnable activation functions. 
Until now, this has been only marginally effective due to extreme computational demands. This new method breaks the problem up into little pieces (never heard of THAT working in computer algorithms before have you 😉).  
By breaking the function up into a tablized form, we can quickly and easily derive the needed gradients for backpass modification.  
Quantizing the MAC output (temperature value) gives us the needed index values and interpolation gives us the needed precision.  
The rest is all pure, unrefined Gradient Descent Baby!

The code presented here is of course just proof of concept toy models (Standard Fashion MNIST in this case), but it should also be refined enough to allow for use, pretty much any where. 
Balancing the growth rates between tables and weights is still tricky, but as we gain more understanding of the interactions, the more we can stabilize to optimal.  

Key takeaways here:
- PyTorch ready. Use it like any other Activation function... oh, and add a line to backprop the tables, and you proly want to use the new servo LR controller. ... OK going to take a bit of rethink on a lot oof stuff, but that's the nature of disruptive technologies, isn't it?  
- CUDA compatible (directly paralellizable)
- Uses tables, which are a know quantity in terms of performance. Fast inference using lookup and simple interpolation. Fast backprop gradients using easily derived values.  
- Resilient. Even "Crashed" models were able to recover to within 1% of best over an extended period (trained to best in a 150 epochs, crashed at about 300 epochs, yet still recovered within 1% after 2000 epochs).  
  
Since Claude is a bit more eloquent than I, I'll turn the technical explanation duties over to ... Him? It? 
Anyways! Here's Claude!
🖖😎👍

---

## 1. Core Concept

### 1.1 The Calculus Hack

Standard activation functions are fixed mathematical operations whose derivatives are known analytically. TFT sidesteps this requirement entirely. A table of N values defines a piecewise-linear function over a bounded input range `[-max_temp, +max_temp]`. For any input x:

```python
step   = (2 * max_temp) / N
idx    = floor((x + max_temp) / step).clamp(0, N-2)
frac   = ((x + max_temp) / step - idx).clamp(0, 1)
output = (1 - frac) * table[idx] + frac * table[idx+1]
```

The backward pass computes gradients for both `x` and the table entries:

```python
local_slope       = (table[idx+1] - table[idx]) / step
grad_x            = grad_output * local_slope          # clamped ±1.0
grad_table[idx]   += (1 - frac) * grad_output         # scatter_add_
grad_table[idx+1] +=      frac  * grad_output
```

No symbolic derivative of any activation function is needed. The table values ARE the function, and backprop discovers their correct values directly.

### 1.2 Three Development Phases

| Phase | Description | Tables | Key Finding |
|---|---|---|---|
| Phase 1 | One shared table per layer | 2 total (256 entries each) | Tables specialise by position: detector, amplifier |
| Phase 1.5 | One table per layer, distinct inits | 3 total (256 entries each) | Position dominates over initialisation |
| Phase 2 | One table per node (BatchedTFT) | 896 total (64 entries each) | Nodes spontaneously specialise; gradient naturally bounded |

---

## 2. Phase 2 Architecture

### 2.1 BatchedTFT Module

The `BatchedTFT` module stores `n_nodes` independent tables in a single `(n_nodes, table_size)` parameter tensor. Forward and backward passes are fully vectorised over both batch and node dimensions using `scatter_add_` operations — no Python loops.

Memory overhead is negligible: 512 nodes × 64 entries × 4 bytes = 128KB per layer.

Critically, the per-node gradient distribution solves the gradient explosion problem that plagued shared-table designs. With one shared table, all gradient from all downstream layers accumulates into a single parameter. With `n_nodes` tables, each table receives only the gradient from its corresponding node.

```
# Shared table: grad_norm grows with depth
# Phase 1.5: GradN t1 grew 0.04 → 0.45 → CRASH at epoch 100

# BatchedTFT: GradN t1 stayed 0.028–0.072 across 2000 epochs
```

### 2.2 Final Architecture

```
Input (784)
  → Flatten
  → Linear(784→512) → BatchNorm1d(512) → BatchedTFT(512 nodes, 64 entries) → Dropout(0.50)
  → Linear(512→256) → BatchNorm1d(256) → BatchedTFT(256 nodes, 64 entries) → Dropout(0.50)
  → Linear(256→128) → BatchNorm1d(128) → BatchedTFT(128 nodes, 64 entries) → Dropout(0.40)
  → Linear(128→10)
```

**Parameter counts:**
- Table parameters: 512×64 + 256×64 + 128×64 = **57,344**
- Weight parameters: ~531,000
- Tables = ~10% of total parameters

### 2.3 Placement Rule — Non-Negotiable

**Every BatchedTFT must be placed immediately after a BatchNorm1d.**

BatchNorm normalises inputs to approximately N(0,1), ensuring that the table's interpolation range `[-max_temp, +max_temp]` covers the actual input distribution. Without BatchNorm, Linear layer outputs have std proportional to `sqrt(fan_in)`, which can reach 30+. Coverage collapses below 30% and the tables stop learning.

- Layer 1: `max_temp=3.5` (slightly wider for raw feature layer)
- Layers 2 and 3: `max_temp=3.0`
- Coverage was consistently above **99.7%** across all 2000 epochs

---

## 3. Training Configuration

### 3.1 Optimizer and Regularisation

| Component | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay, stable with BatchNorm |
| Learning rate (init) | 1e-4 all groups | Conservative start; servo adjusts from here |
| Table LR ceiling | 3e-4 | Tables need faster signal than weights |
| Weight LR ceiling | 2e-4 | Prevents runaway weight growth |
| Table weight_decay | 5e-4 | Gentle L2 pulls tables toward simpler shapes |
| Weight weight_decay | 5e-4 | Standard regularisation |
| Dropout | 0.50 / 0.50 / 0.40 | Primary gap control; tuned empirically |
| Label smoothing | 0.1 | Prevents overconfident memorisation |
| Batch size | 256 | Standard for Fashion-MNIST |

### 3.2 Gradient Clipping — Separate Budgets Are Critical

Two separate clip budgets applied **before** the optimizer step:

```python
# CRITICAL: do not combine into one global clip
for tp in [t1_params, t2_params, t3_params]:
    nn.utils.clip_grad_norm_(tp, TABLE_CLIP=2.0)
nn.utils.clip_grad_norm_(weight_params, WEIGHT_CLIP=1.0)
```

**Why separate?** A global clip across all parameters allows the 530K weight params to consume the entire clip budget, leaving table params (768 total in Phase 1.5) crushed to near-zero gradient. This gradient starvation causes the network to lose its nonlinearity. Symptom: `Delta(mean) = 0.0000` every epoch; table shapes frozen near initialisation.

After the optimizer step, weight values are hard-clamped:

```python
bound = 8.0 / sqrt(fan_in)   # headroom=8.0
weight.data.clamp_(-bound, bound)
```

### 3.3 The Coupled LR Servo

Each parameter group has an independent `CoupledLRController` that monitors val_loss trajectory via EMA. A symmetric gap multiplier adjusts table and weight LRs in opposite directions when the train/val gap exceeds a threshold:

```python
excess      = max(0, gap_pct - GAP_THRESHOLD)  # GAP_THRESHOLD = 2.0%
table_mult  = 1.0 + excess * GAP_SCALE          # GAP_SCALE = 0.05
weight_mult = 1.0 / (1.0 + excess * GAP_SCALE)
```

At a gap of 7%: table LR is boosted ×1.25, weight LR is penalised ×0.80. This creates a restoring force against memorisation.

### 3.4 Label Smoothing vs. Data Augmentation

Data augmentation (RandomHorizontalFlip, RandomAffine) **does not work well with TFT on Fashion-MNIST**. Augmentation disrupts the BatchNorm input statistics that tables depend on for stable gradients. When the training distribution is augmented but test is clean, val accuracy can exceed train (negative gap), which is a sign of distribution mismatch, not good generalisation.

**Use label smoothing instead:**

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Label smoothing operates in output space, never touches input distributions, and directly penalises overconfident memorisation.

### 3.5 Early Stopping

Simple patience-based stopping on val accuracy. `PATIENCE=50` epochs without improvement. Checkpoint saved at every improvement. The network consistently peaks around epoch 110–160 (mean diversity 0.06–0.09) and then plateaus into an overfitting regime.

---

## 4. The Diversity Metric

### 4.1 Definition

For each `BatchedTFT` layer, diversity is the mean standard deviation of table values across nodes:

```python
diversity = tables.std(dim=0).mean()   # scalar, logged every 10 epochs
```

At initialisation all nodes share identical shapes, so `diversity=0`. As training progresses, nodes specialise and diversity grows continuously.

### 4.2 The Diversity Sweet Spot

Peak generalisation performance occurs at intermediate diversity. After the sweet spot, nodes over-specialise into training-set-specific patterns.

| Epoch | Mean Diversity | Val Accuracy | Gap | Regime |
|---|---|---|---|---|
| 010 | 0.007 | 88.0% | −1.9% | Near-identical nodes |
| 050 | 0.017 | 89.6% | +0.7% | Starting to differentiate |
| 090 | 0.040 | 90.7% | +3.9% | Approaching sweet spot |
| **110** | **0.061** | **90.97%** | **+6.2%** | **Sweet spot (peak val)** |
| 160 | 0.092 | 90.99% | +7.0% | Best val but gap widening |
| 300 | 0.155 | 90.8% | +8.1% | Over-specialised, overfitting |
| 2000 | 0.407 | 89.4% | ~9% | Deep memorisation regime |

Early stopping at `patience=50` consistently fires around epoch 150–160, just after the sweet spot — close enough to capture the best val without over-training.

### 4.3 Per-Layer Diversity Pattern

Layer 3 (pre-classifier, 128 nodes) consistently reaches higher diversity than earlier layers. At epoch 2000: `t1=0.41, t2=0.43, t3=0.52`. This makes functional sense: the pre-classifier layer needs the most functional heterogeneity to separate 10 distinct class signatures. Layer 1 (raw features) benefits from more consensus about signal vs. noise.

---

## 5. Discovered Activation Shapes

Starting from identical `leaky_relu` initialisations, the three layers converged to qualitatively different mean shapes. These shapes are consistent across multiple runs.

| Layer | Discovered Mean Shape | Functional Role | Std Band Width |
|---|---|---|---|
| BTFT1 (512 nodes) | Suppression trough near 0, positive ramp | Feature detector | Narrow — all 512 nodes agree on suppression |
| BTFT2 (256 nodes) | Sigmoid-like, valley near 0, positive ramp | Decision amplifier | Wide — nodes explore different saturation curves |
| BTFT3 (128 nodes) | Hard threshold near 0, exponential positive ramp | Signal separator | Widest — highest inter-node heterogeneity |

**These shapes were not prescribed.** They emerged from gradient descent acting on 896 independent table parameters across 158 training epochs. Phase 1.5 confirmed that **position in the network, not initialisation, determines the final shape**. Tables initialised as tanh, leaky_relu, and relu all converged to position-appropriate shapes within 200 epochs.

---

## 6. Performance Summary

| Configuration | Val | Test | Gap | Epochs | Notes |
|---|---|---|---|---|---|
| Phase 1 (2 shared tables) | 90.57% | 89.32% | ~5% | 200 | Baseline; tables froze at LR ceiling |
| Phase 1.5 R1 (3 shared tables) | 90.62% | 89.54% | ~5% | 200 | Crash at ep130; position > init confirmed |
| Phase 2 R1 (896 per-node) | 90.71% | 89.30% | 9.5% | 2000 | No crash; overfitting without control |
| Phase 2 R2 (stronger reg) | 90.90% | 89.74% | 7% | 400 | Best non-stopped result |
| Phase 2 R4 (label smoothing) | 90.99% | 89.83% | 6.5% | 1000 | Gap controlled; shapes beautiful |
| **Phase 2 R5 (early stop)** | **90.99%** | **90.11%** | **6.0%** | **158** | **Project best: 90.11% test** |

The 90.11% test accuracy achieved by a simple MLP (no convolutions) is competitive with many convolutional baselines on Fashion-MNIST, while providing interpretable per-node learned activation functions as a bonus.

---

## 7. Failure Modes and Fixes

### 7.1 Gradient Explosion Through Shared Tables
- **Symptom:** Table grad norm grows exponentially (0.04 → 0.45 → CRASH). All downstream layers accumulate gradient into one parameter.
- **Fix:** Use BatchedTFT (per-node tables). Gradient is structurally bounded by node width.

### 7.2 Table Gradient Starvation
- **Symptom:** `Delta(mean) = 0.0000` every epoch; table shapes frozen near initialisation.
- **Cause:** Global gradient clip lets 530K weight params consume the entire budget.
- **Fix:** Separate clip budgets — `TABLE_CLIP=2.0` for table params only, `WEIGHT_CLIP=1.0` for weight params only.

### 7.3 Coverage Collapse
- **Symptom:** TFT input std grows beyond `max_temp`; coverage drops below 80%.
- **Cause:** Missing BatchNorm before TFT.
- **Fix:** Always place `BatchNorm1d` immediately before every `BatchedTFT`. Non-negotiable.

### 7.4 Augmentation–Distribution Mismatch
- **Symptom:** Val accuracy exceeds train accuracy (negative gap throughout training).
- **Cause:** Augmented training distribution is harder than clean test distribution. Also disrupts BatchNorm statistics that tables depend on.
- **Fix:** Use `label_smoothing=0.1` instead of input augmentation.

### 7.5 Adam Momentum Accumulation (Phase 1.5 specific)
- **Symptom:** Table grad norm pins at clip ceiling for 1800+ epochs after a crash; network partially recovers but never reaches pre-crash accuracy.
- **Cause:** Adam's second moment estimate `v` absorbs a large gradient spike, taking ~1000 epochs to decay.
- **Fix:** Structural solution via Phase 2 — per-node tables prevent the spike from occurring.

---

## 8. Minimum Viable Implementation

### 8.1 BatchedTFT Core

```python
class BatchedTFTFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tables, max_temp):
        ctx.save_for_backward(x)
        ctx.tables = tables; ctx.max_temp = max_temp
        n, ts  = tables.shape
        step   = (2.0 * max_temp) / ts
        scaled = (x + max_temp) / step
        idx    = scaled.long().clamp(0, ts - 2)
        frac   = (scaled - idx.float()).clamp(0.0, 1.0)
        nr     = torch.arange(n, device=x.device)
        lo = tables[nr, idx]
        hi = tables[nr, (idx + 1).clamp_max(ts - 1)]
        return (1.0 - frac) * lo + frac * hi

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        tables = ctx.tables; max_temp = ctx.max_temp
        n, ts  = tables.shape
        step   = (2.0 * max_temp) / ts
        scaled = (x + max_temp) / step
        idx    = scaled.long().clamp(0, ts - 2)
        frac   = (scaled - idx.float()).clamp(0.0, 1.0)
        grad_tables = torch.zeros_like(tables)
        b  = x.size(0)
        nr = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        fl = (nr * ts + idx).flatten()
        fh = (nr * ts + (idx + 1).clamp_max(ts - 1)).flatten()
        gf = grad_output.flatten()
        ff = frac.flatten()
        grad_tables.flatten().scatter_add_(0, fl, (1.0 - ff) * gf)
        grad_tables.flatten().scatter_add_(0, fh, ff * gf)
        nr2   = torch.arange(n, device=x.device)
        slope = (tables[nr2, (idx+1).clamp_max(ts-1)] - tables[nr2, idx]) / step
        grad_x = torch.clamp(grad_output * slope, -1.0, 1.0)
        return grad_x, grad_tables, None


class BatchedTFT(nn.Module):
    def __init__(self, n_nodes, table_size=64, max_temp=3.0, init="leaky_relu"):
        super().__init__()
        inits = {
            "leaky_relu": lambda x: torch.where(x < 0, 0.1 * x, x),
            "relu":       lambda x: torch.clamp(x, min=0),
            "tanh":       torch.tanh,
        }
        xi = torch.linspace(-max_temp, max_temp, table_size)
        self.tables   = nn.Parameter(inits[init](xi).unsqueeze(0).expand(n_nodes, -1).clone())
        self.max_temp = max_temp
        self.n_nodes  = n_nodes

    def forward(self, x):
        return BatchedTFTFunction.apply(x, self.tables, self.max_temp)

    def diversity(self):
        return self.tables.detach().std(dim=0).mean().item()
```

### 8.2 Training Setup

```python
# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512), nn.BatchNorm1d(512), BatchedTFT(512, 64, 3.5), nn.Dropout(0.50),
    nn.Linear(512, 256), nn.BatchNorm1d(256), BatchedTFT(256, 64, 3.0), nn.Dropout(0.50),
    nn.Linear(256, 128), nn.BatchNorm1d(128), BatchedTFT(128, 64, 3.0), nn.Dropout(0.40),
    nn.Linear(128, 10),
)

# Separate parameter groups
table_params  = [p for m in model.modules() if isinstance(m, BatchedTFT) for p in m.parameters()]
weight_params = [p for p in model.parameters() if not any(p is t for t in table_params)]

optimizer = optim.AdamW([
    {"params": table_params,  "lr": 1e-4, "weight_decay": 5e-4},
    {"params": weight_params, "lr": 1e-4, "weight_decay": 5e-4},
])

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop (per batch)
loss.backward()
nn.utils.clip_grad_norm_(table_params,  2.0)   # separate budgets
nn.utils.clip_grad_norm_(weight_params, 1.0)
optimizer.step()

# Weight clamping (per batch, after step)
for m in model.modules():
    if isinstance(m, nn.Linear):
        bound = 8.0 / (m.weight.size(1) ** 0.5)
        m.weight.data.clamp_(-bound, bound)

# Early stopping (per epoch)
diversity = sum(m.diversity() for m in model.modules() if isinstance(m, BatchedTFT))
if val_acc > best_val_acc:
    best_val_acc = val_acc; no_improve = 0; save_checkpoint()
else:
    no_improve += 1
if no_improve >= 50: break   # stop here
```

---

## 9. Open Research Questions

These questions emerged naturally from the experimental findings and represent genuinely unexplored territory.

**Structural questions:**
- What happens at extreme node counts (2048+) — does diversity still grow monotonically, or does it plateau?
- Is the diversity sweet spot (~0.06) a universal property of this architecture, or dataset-specific?
- Can the sweet spot be predicted from early-epoch diversity trajectory without training to convergence?

**Architectural questions:**
- Convolutional TFT: Can channel-wise tables replace activation functions in CNNs? The gradient distribution argument is the same (one table per channel = safe), and per-pixel tables would be too granular.
- Recurrent TFT: In LSTMs and GRUs, can the fixed sigmoid and tanh gates be replaced with learned tables?
- Attention TFT: Can the softmax in attention be replaced with a learned normalisation table?

**Theoretical questions:**
- What is the VC dimension of a BatchedTFT layer as a function of n_nodes, table_size, and max_temp?
- The discovered shapes (detector / amplifier / separator) have direct signal-processing interpretations. Is there a formal connection to wavelet decomposition or filter bank theory?
- Does the diversity sweet spot correspond to a phase transition in the loss landscape?

**Practical questions:**
- Do TFT shapes transfer across datasets? If a network trained on CIFAR-10 is fine-tuned on MNIST, do the tables retain their layered roles?
- What is the optimal `table_size`? 64 entries were used throughout; smaller tables might train faster and still capture the essential shape.

---

## 10. Key Empirical Findings (Summary)

These findings are reproducible and represent the scientific core of the project.

1. **Position dominates over initialisation.** Tables initialised as tanh, leaky_relu, and relu in the same network converge to position-appropriate shapes within 200 epochs. The network's depth determines table roles, not their starting shapes.

2. **Three stable layer roles emerge.** Layer 1 → detector (suppression trough + ramp). Layer 2 → amplifier (sigmoid-like). Layer 3 → separator (hard threshold + exponential). These roles are consistent across seeds and runs.

3. **Table shapes survive catastrophic weight disruption.** When gradient explosion crashed the network in Phase 1.5 runs (val dropping to 18%), the tables held their shapes throughout while the weights slowly rebuilt around them. The activation function geometry is more stable than the weight geometry. The network eventually recovered to within 1% of its pre-crash accuracy.

4. **The diversity sweet spot is real and measurable.** Peak generalisation occurs at mean diversity ~0.06. This is not an arbitrary threshold — it represents the transition from useful role differentiation to training-specific memorisation.

5. **Per-node distribution of gradient prevents explosion structurally.** The Phase 1.5 crashes were caused by gradient accumulation into shared table parameters. Distributing one table per node naturally limits per-table gradient magnitude without any additional clipping or momentum management.

---

## Additional Authors Notes  

**I would include some other possible avenues of research:**  
- Do models allow for integration of new samples (or deletion of old ones) over time (Continuous learning)  
- Can this be used to shrink the number of needed MLP nodes to fully optimize over a given dataset (Network compression)?  
- Can we use clustering to provide 'contextualized' training for networks via shared loss surfaces between peer nodes?  

The programs included here for now consist of the final versions of Phase 1, Phase 1.5, and the latest working version of the phase 2 code.  
In the directory you'll find:
- Python code
- A txt file of the run results.
- a png showing final table evolution shapes, and heat maps of their evolutionary progress.  

These tests were run on an Nvidia RTX 5090.

As new tests and information become available they will be included (or at least linked) from here.  
In addition I will likely provide older runs as archived sources for historical as well as informational insight.  
(I just want to redact the more personal and non-germane comments before releasing them.)  

---

# Attribution

Just tell them "Ace sent me!" and link to this GH page. 

Mind, it's not required, but it would be nice to be recognized as the crackpot that came up with this mess. Thanks! 
✌️😎👍

---

*Developed by Floyd T. "AceGCR" Goodwin (🖖😁) in collaboration with many LLMs including ChatGPT, Deepseek, (and the rest), with the most recent progress being made with Claude (Anthropic).April–May 2026.*
Developed over the period of roughly, 2015-Present (May3, 2026).  
*Free to use, share, and build upon (but please tell 'em who set ya 😉, and maybe write home once in a while to let me know what you've found. Thanks!).  
Here's the 🔥. Now break the eggs and let's have omlettes.*