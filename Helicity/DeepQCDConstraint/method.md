# Prompt: Deep-QCD methodology diagram

Copy everything below the line into a diagram or image-generation tool, or give it to
a designer. Every name, shape and number in it matches the Deep-QCD code.

---

Create a publication-quality methodology figure for a physics paper. It must explain,
on its own, how a physics-constrained deep neural network (DNN) learns lattice QCD
matrix elements M(ω, P) from synthetic data, extrapolates them to large Ioffe time ω,
and gives the Ioffe-time distribution I(ω) and its continuum limit.

Part A describes the method in full, so that every label is correct. Part B says
exactly what to draw.

## Part A: the method

### A1. Notation

- **P**: momentum index, P ∈ {1, 2, 3, 4, 5}.
- **ω**: Ioffe time, from 0 to ω_max = 20.
- **FM**: lattice ensemble. FM 9, FM 12 and FM 15 have lattice spacings
  a = 0.0888, 0.1207 and 0.151 fm.
- **M(ω, P)**: the matrix element the network predicts.
- **I(ω)**: the Ioffe-time distribution.
- **Momentum-pair relation.** For momenta i < j,
  I_ij(ω) = (j²M_j − i²M_i)/(j² − i²).
- **Physics constraint.** For i < j at the same ω: M_i/M_j < j²/i², which is the same
  as i²M_i < j²M_j, and the same as every I_ij > 0.

### A2. Synthetic data generation

Real lattice data are sparse: 3–5 ω values for each (FM, P). The network is
therefore trained only on synthetic data, which are built from the real data in
four steps, separately for each FM.

**G1. Real-data preparation.**

- The real data are jackknife samples, indexed by Exp; there are 1026, 1013 and
  900 of them for FM 9, 12 and 15.
- Each real point sits at ω = P·z·d₁.
- For each (FM, P), keep the smallest ω values: 5 for FM 9, 4 for FM 12 and 3 for
  FM 15. Add the anchor M(ω = 0) = 0.
- For each (P, ω) cell, compute the mean over Exp and its jackknife error.

**G2. Model selection and the reference curve.**

- Fit eight candidate forms of M(ω, P) to the cell means, for P = 1…5 and ω > 0.
  The fits are weighted least squares with weight 1/σ²_JK and use multiple starts.
- Rank the forms by QAICc: χ² is divided by the overdispersion ĉ (the χ²/dof of
  the most flexible form), and ĉ counts as one extra parameter. Check the ranking
  with an Exp bootstrap.
- The selected form is M(ω, P) = A·ω·e^{−sω}·P^{−γ}, with A, s and γ fitted for
  each FM.
- The fitted curve at P = 3 becomes the reference M₃(ω) on the grid
  ω = 0, 0.1, …, 20 (201 points).

**G3. Physics-consistent noise-free curves.**

All momenta are derived from M₃ so that every momentum pair gives the same I(ω),
under M_P = I + Y/P²:

- Solve for M₁ and M₂ from M₃ = (1 − C)·M₁ + C·M₂, where C follows from the
  pair relation. Start halfway to the I = 0 point (M_P = 9M₃/(2P²)) and use
  gradient descent.
- Then I(ω) = (9M₃ − M₁)/8, Y = M₁ − I, and
  M_P = I + (M₁ − I)/P² for P = 4, 5.
- The result is Est_M(ω, P) = I(ω) + Y(ω)/P². It satisfies the pair relation
  exactly, so it can be the noise-free truth.

**G4. Noisy replicas.**

- For each (FM, P), fit a straight line std(ω) = slope·ω + intercept to the spread
  of the real M over Exp at the measured ω values. Extrapolate it to ω = 20,
  floored at a tiny positive value.
- For each (FM, P, ω), draw N_FM replicas from a normal distribution with mean
  Est_M and that std, truncated at M ≥ 0. N_FM is 1026, 1013 or 900, the same as
  the real Exp count.
- Smooth each replica curve along ω with a Savitzky–Golay filter (window 60
  points, order 3), and set M(ω = 0) = 0.
- Result for each FM: a tensor M[P, Exp, ω] with shape (5, N_FM, 201), plus
  Est_M[P, ω].
- For training, replica number e of all five P is taken together as one joint
  sample.

### A3. One training example (a "window")

A window has, for each of the 5 momenta, 4 known seed points (ω, M). It also has
6 target ω values that are the same for all P. Two seed layouts are used, 50 % each:

| Layout | Seed ω values | Target ω values |
| --- | --- | --- |
| grid | ω₀ + kΔω, k = 0…3, the same for every P | ω₀ + kΔω, k = 4…9 |
| lattice | P·z·d₁, z = 0…3, with d₁ ~ U[0.15, 0.45] | Δω·k, k = 1…6 |

- **Grid layout:** the start ω₀ is random. Every rollout block after the first one
  looks like this.
- **Lattice layout:** each P has its own spacing P·d₁. This mimics real lattice data,
  which have d₁ = 0.196, 0.262 and 0.393.
- **Step size:** Δω = 0.1 × stride, with stride ∈ {1, 2, 5, 10}, so
  Δω ∈ {0.1, 0.2, 0.5, 1.0}.
- **Values:** M at off-grid ω values comes from linear interpolation of the replica
  curve.
- **Supervision:** the replica M of every P at the 6 targets, shape (5, 6).
- Fresh random windows are drawn at every training step.

### A4. Network input (46 numbers per window)

1. **Seed scale:** s = the largest seed M over all P and all seeds of the window.
2. **Input vector**, in this order:
   - 20 seed ω values divided by ω_max (P-major, 5 × 4);
   - 20 seed M values divided by s (5 × 4);
   - 6 target ω values divided by ω_max.
3. **Standardisation:** each of the 46 features is standardised with its mean and
   standard deviation, measured on 100 000 training windows.

Deliberately not given to the network: the lattice spacing a² and the absolute level
of M. Without them the network has to read the shape of the seeds, so it can work for
an ensemble it has never seen.

### A5. Network and physics-constraint layer

- **MLP:** Linear(46 → 256) + SiLU, repeated 3 times, then Linear(256 → 12).
- **Output scaling:** the 12 outputs are rescaled with a fixed median and IQR taken
  from the noise-free curves, clamped to [−30, 30], and reshaped to 6 × 2. For each
  target ω this gives (u₁, u₂) = (log M₁/s, log I/s).
- **Constraint layer** (fixed, no learned weights):
  - M₁ = s·e^{u₁} > 0 and I = s·e^{u₂} > 0
  - M_P = I + (M₁ − I)/P², for P = 1…5
- **What this guarantees, by construction:**
  - P²M_P = M₁ + (P² − 1)·I rises strictly with P, so the constraint holds for
    every pair.
  - M_P > 0 for every P.
  - Every momentum pair gives exactly the same I(ω).
- **Output:** M with shape (5 P × 6 targets) and I with shape (6).

### A6. Loss and training

- **Loss:** L = mean over P and the 6 targets of [(M̂_P − M_P^replica)/w_P]².
  - w_P is the largest seed of that P, with a floor of 10⁻³·s.
  - This weighting makes P = 5 count as much as P = 1.
- **Optimiser and batches:**
  - Adam, learning rate 10⁻³; 4096 windows per batch, 200 batches per epoch.
  - ReduceLROnPlateau (factor 0.5, patience 6); at most 300 epochs; early stopping
    after 20 epochs without improvement. The checkpoint with the best validation
    loss is kept.
- **Validation:** 10 % of each training FM's replicas are held out, giving 100 000
  validation windows.
- **Leave-one-FM-out:**
  - Model FM9-out trains on FM 12 + 15 and is tested on FM 9.
  - Model FM12-out trains on FM 9 + 15 and is tested on FM 12.
  - Model FM15-out trains on FM 9 + 12 and is tested on FM 15.
  - Model all-FM trains on all three FMs and is used for the real data.

### A7. Autoregressive rollout (inference)

The output grid is ω = kΔω, from k = 0 up to ω = 20, with M(0) = 0.

1. **First block.**
   - Grid start: seeds at k = 0…3 for every P.
   - Lattice start: seeds at each P's own lattice ω; the network predicts the grid
     points k = 1…6.
2. **Repeat:** take the last 4 known grid points of all 5 P as seeds and predict the
   next 6. The final block is shifted back so that it ends exactly at ω = 20.
3. **I(ω):** at every predicted point, I(ω) is the network's own I, and every
   momentum pair gives the same value.
4. **Errors:** the rollout is run on every replica (Exp). Errors are jackknife
   errors: σ² = (N − 1)/N · Σᵢ(θᵢ − θ̄)².

### A8. Evaluation (synthetic, held-out FM)

- **Short-range test:** windows (4 seeds → 6 targets) in both layouts and at all
  Δω. Score: RMSE against the noise-free Est_M for each P.
- **Full rollout:** to ω = 20 from every replica. Scores: RMSE of the replica-mean
  M and of I(ω).
- **Constraint-violation rate:** the fraction of (sample, ω) points where P²M_P does
  not rise strictly with P.
- **Baseline:** the unconstrained DeepQCD model, which predicts each P separately.

### A9. Application to real lattice data

- **Input:** the real lattice matrix elements, one value per (FM, P, Exp, Z).
  - Exp indexes jackknife samples; they come from the same gauge configurations for
    every P.
  - The real ω values are ω = P·z·d₁.
- **Seeds:** Z = 0…3 for each P, which is the lattice layout, fed to Model all-FM and
  rolled out to ω = 20.
- **Check points:** real points beyond the seeds are never given to the model; they
  are used to check it (FM 9: Z = 4, 5; FM 12: Z = 4).

### A10. Continuum limit a → 0

At each ω, fit the three ensembles:

- **Main fit:** I(ω, a) = I₀(ω) + c(ω)·a².
- **Systematic check:** I(ω, a) = I₀(ω) + c(ω)·a.
- **Statistical error:** the ensembles are independent and the fit is linear, so
  σ²_stat = Σ_FM w_FM²·σ²_JK,FM, where w_FM are the fit weights.
- **Systematic and total error:** σ_sys = |I₀(a²) − I₀(a)|, and
  σ_tot = σ_stat ⊕ σ_sys (added in quadrature).
- **Applied to:** the synthetic held-out rollouts, as a check against the
  extrapolated Est_M, and the real-data rollouts.
- **Final output:** the continuum I(ω) with error bands.

## Part B: what to draw

### Output specifications

- Resolution: 300 dpi.
- Size: 7.2 in × 6.6 in (2160 × 1980 px), for a full-width figure.
- Also export the diagram as vector PDF or SVG with the same layout.
- White background, flat style: no 3D, no shadows, no gradients, no clip art, no
  logos.
- Font: sans-serif such as Helvetica or Arial. Box titles 9 pt bold, body text
  7–7.5 pt, math in italic serif in LaTeX style with true subscripts and
  superscripts (M₁, P², a²). Every text element must still be readable when the
  figure is printed at 7.2 in wide.
- Use a colourblind-safe palette, one colour per stage:
  - synthetic data generation: teal #0f8f9a
  - data: blue #2a78d6
  - inputs and network: orange #eb6834
  - training: green #1baf7a
  - inference and evaluation: purple #8e5bd0
  - continuum: dark grey #555555

  Boxes have rounded corners, a 1 pt outline in the stage colour and a 10 % tint
  of the same colour as fill.
- Arrows: 1 pt, dark grey #333333, with small arrowheads. Label an arrow with the
  tensor shape it carries where one is given below.
- Label the generation steps "G1"–"G4" in small teal rounded tags, and every
  other box with a numbered circle ①–⑨, each at its top-left corner.
- Use only the text given here. Do not add result values, a figure title or
  extra commentary.

### Layout: four rows, read left to right and top to bottom

#### Row 0: synthetic data generation (a full-width teal band, about 25 % of the figure height)

Draw a band titled "Synthetic data generation (per FM)" containing four boxes of
equal width, joined by arrows G1 → G2 → G3 → G4.

**G1. Real lattice data.**

- Mini plot: sparse points with error bars for 5 P, at a few small ω values each.
- Text:
  - "jackknife samples Exp: 1026 / 1013 / 900"
  - "ω = P·z·d₁; keep the 5 / 4 / 3 smallest ω (FM 9 / 12 / 15)"
  - "anchor M(0) = 0"
  - "cell mean ± σ_JK"

**G2. Model selection (QAICc).**

- Mini graphic: a small ranked bar list of 8 forms with the top one highlighted.
- Text:
  - "8 candidate forms of M(ω, P), weighted fit 1/σ²_JK"
  - "QAICc = χ²/ĉ + 2k′ (+ small-N term), bootstrap check"
  - "selected: M = A·ω·e^{−sω}·P^{−γ}"
  - "reference curve: M₃(ω) at P = 3, ω = 0…20, Δω = 0.1"

**G3. Pair-consistent curves.**

- Mini plot: 5 smooth M(ω) curves for P = 1…5 and one I(ω) curve beneath them.
- Text:
  - "solve M₁, M₂ from M₃ = (1 − C)M₁ + C M₂"
  - "I = (9M₃ − M₁)/8"
  - "M_P = I + (M₁ − I)/P², P = 4, 5"
  - "Est_M = I + Y/P²: every pair gives the same I"

**G4. Noisy replicas.**

- Mini plot: one Est_M curve with a band that widens with ω and a few thin wiggly
  replica lines.
- Text:
  - "std(ω) = slope·ω + intercept (fit to real spread, extrapolated)"
  - "N_FM draws ~ Normal(Est_M, std), truncated at M ≥ 0"
  - "Savitzky–Golay smoothing (window 60, order 3), M(0) = 0"
  - "M[P, Exp, ω]: (5, N_FM, 201)"

Draw a thick arrow from G4 down to box ①, labelled "synthetic training set".

#### Row 1: data → inputs → network

**① Synthetic training set (blue).**

- A mini plot of 5 coloured M(ω) curves (P = 1…5) that rise and then fall, with a
  thin noise band around each.
- Text:
  - "Est_M(ω,P) = I(ω) + Y(ω)/P² (noise-free truth)"
  - "Replicas Exp: 1026 / 1013 / 900 (FM 9 / 12 / 15)"
  - "replica e of all P = one joint sample"
  - "tensor M[P, Exp, ω]: (5, N_FM, 201)"

**② Window sampling (blue).** Two mini panels side by side, each showing 5 rows of
dots (P = 1…5) along an ω axis, with filled dots for seeds and hollow dots for
targets.

- **"grid layout":** the 4 seeds sit at the same ω for every P, "ω₀ + kΔω,
  k = 0…3", and the 6 targets follow at k = 4…9.
- **"lattice layout":** the seeds spread out as P grows, "ω = P·z·d₁, z = 0…3,
  d₁ ~ U[0.15, 0.45]", and the 6 targets are shared, "Δω·(1…6)".
- Caption under the panels: "Δω ∈ {0.1, 0.2, 0.5, 1.0} · layouts 50 / 50 ·
  replica e of all P together".

**③ Input vector (orange).** Draw a horizontal bar split into three coloured
segments, with their widths in the ratio 20 : 20 : 6.

- Segment labels:
  - "seed ω / ω_max (5×4)"
  - "seed M / s (5×4)"
  - "target ω / ω_max (6)"
- Below the bar:
  - "s = max seed M (all P)"
  - "standardised → 46 features"
  - in small italics: "no a², no absolute level"
- Label the outgoing arrow "46".

**④ Constrained joint-P network (orange). This is the largest box and the visual
centrepiece.**

- Draw the flow left to right:
  - three stacked dense layers labelled "Linear 256 + SiLU" (×3);
  - then "Linear → 12";
  - then a small box "rescale (median/IQR), clamp, reshape 6×2";
  - then two output channels, "u₁ = log M₁/s" and "u₂ = log I/s".
- Then a highlighted sub-box with a thicker outline, titled "Physics-constraint
  layer (fixed)", containing:
  - "M₁ = s·e^{u₁},  I = s·e^{u₂}"
  - "M_P = I + (M₁ − I)/P²"
- Below the sub-box, three guarantees with check marks:
  - "✓ P²M_P = M₁ + (P² − 1)·I increases with P ⇒ M_i/M_j < j²/i²"
  - "✓ M_P > 0 for all P"
  - "✓ every momentum pair gives the same I(ω)"
- Label the output arrow "M: (5×6), I: (6)".

#### Row 2: training → validation → rollout

**⑤ Loss and optimisation (green).**

- "L = mean_{P,targets} [(M̂_P − M_P^replica)/w_P]²"
- "w_P = largest seed of P (floor 10⁻³·s)"
- "Adam 10⁻³ · batch 4096 · 200 steps/epoch"
- "ReduceLROnPlateau · early stop 20 · ≤ 300 epochs"
- "fresh random windows every step"

Draw a curved feedback arrow from ⑤ back to ④, labelled "back-propagation".

**⑥ Leave-one-FM-out validation (green).** A small 4-row table of model icons:

| Model | Train | Test |
| --- | --- | --- |
| Model FM9-out | FM 12 + 15 | FM 9 |
| Model FM12-out | FM 9 + 15 | FM 12 |
| Model FM15-out | FM 9 + 12 | FM 15 |
| Model all-FM | FM 9 + 12 + 15 | real data |

- Highlight the Model all-FM row in purple and send an arrow from it to ⑧.
- Footnote: "10 % of replicas held out for early stopping".

**⑦ Autoregressive rollout and evaluation (purple).**

- Draw an ω axis from 0 to 20 with a chain of overlapping blocks:
  - the first block is labelled "4 seeds → 6";
  - the next blocks are labelled "last 4 → next 6";
  - the last block is labelled "shifted to end at ω = 20".
- Show the seed points as filled dots and the predicted points as hollow dots.
- Text:
  - "grid start or lattice start (first block → k = 1…6)"
  - "every replica → jackknife errors"
  - "held-out FM vs Est_M and vs unconstrained per-P DeepQCD"
  - "metrics: RMSE(M), RMSE(I), constraint-violation rate"

#### Row 3: real data → continuum

**⑧ Real lattice data (purple).**

- A mini plot of real data points with error bars: filled circles for the seeds and
  hollow squares for the check points, with DNN curves running through.
- Text:
  - "real lattice M: jackknife samples Exp (same configurations for all P)"
  - "ω = P·z·d₁, d₁ = 0.196 / 0.262 / 0.393"
  - "seeds Z = 0…3 per P → Model all-FM → rollout to ω = 20"
  - "check points never fed: FM 9 Z = 4, 5; FM 12 Z = 4"
- Outputs: "M(ω,P) ± σ_JK" and "I(ω) ± σ_JK" for each FM.

**⑨ Continuum extrapolation a → 0 (dark grey).**

- An inset plot with I at a fixed ω on the y-axis and a² on the x-axis. Show three
  points at a = 0.0888, 0.1207 and 0.151 fm, a solid fit line extended to a = 0, and
  a dashed line for the linear-in-a check.
- Text:
  - "pointwise in ω: I(ω,a) = I₀(ω) + c(ω)·a²"
  - "systematic: I₀ + c·a"
  - "σ²_stat = Σ w_FM²·σ²_JK,FM"
  - "σ_sys = |I₀(a²) − I₀(a)|"
  - "σ_tot = σ_stat ⊕ σ_sys"
- Final output in bold: "Continuum I(ω) ± σ_tot".

### Connections

- G1 → G2 → G3 → G4 (row 0), then a thick arrow G4 → ① labelled "synthetic training set".
- A thin dashed arrow from G1 to ⑧ labelled "same real data".
- ① → ② → ③ → ④ (row 1). Label ② → ③ "window: seeds (5×4 ω, M), 6 target ω".
- ④ → ⑤, plus the feedback arrow ⑤ → ④ labelled "back-propagation".
- ⑤ → ⑥ → ⑦ (row 2). Label ⑥ → ⑦ "trained checkpoints".
- ⑥ (the Model all-FM row) → ⑧ → ⑨ (row 3).
- ⑦ → ⑨: a thin dashed arrow labelled "synthetic check: a → 0 vs extrapolated
  Est_M".

### Style notes

- Align the boxes on a grid, keep even spacing and generous white space, and make
  all boxes in a row the same height.
- Use the same per-P colours (P = 1…5) in every mini plot: blue, orange, green,
  red, purple.
- Keep the mini plots schematic: no axis tick numbers, only the axis labels ω, M,
  I and a².
