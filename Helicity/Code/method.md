# Prompt: per-P machine-learning pipeline, methodology diagram

Copy everything below the line into a diagram or image-generation tool, or give it to
a designer. Every name, shape and number in it matches the per-P machine-learning
pipeline code.

---

Create a publication-quality, high-resolution methodology figure for a physics paper.
On its own, it must explain how sparse lattice QCD data are turned into a large
synthetic training set, how five machine-learning regressors (one per momentum P)
learn to step the matrix element M(ω, P) forward in Ioffe time ω, how the trained
models extrapolate M to large ω, and how M gives the Ioffe-time distribution I(ω)
and its continuum limit a → 0.

Part A describes the method in full, so that every label is correct. Part B says
exactly what to draw.

## Part A: the method

### A1. Notation

- **P**: momentum index, P ∈ {1, 2, 3, 4, 5}.
- **ω**: Ioffe time, from 0 to ω_max = 20, on a working grid with step 0.1.
- **FM**: lattice ensemble. FM 9, FM 12 and FM 15 have lattice spacings
  a = 0.0888, 0.1207 and 0.151 fm.
- **Exp**: jackknife sample (real data) or replica (synthetic data).
- **M(ω, P)**: the matrix element.
- **I(ω)**: the Ioffe-time distribution.
- **Momentum-pair relation:** for 2 ≤ p₁ < p₂,
  I(ω) = (p₂²·M(p₂, ω) − p₁²·M(p₁, ω)) / (p₂² − p₁²).

### A2. Synthetic data generation (per FM)

Real lattice data are sparse: 3–5 ω values for each (FM, P). The models are
therefore trained only on synthetic data, which are built from the real data in
four steps.

**G1. Real-data preparation.**

- The real data are jackknife samples: 1026, 1013 and 900 for FM 9, 12 and 15.
- Each real point sits at ω = P·z·d₁, with d₁ = 0.196, 0.262 and 0.393.
- For each (FM, P), keep the 5, 4 and 3 smallest ω values (FM 9, 12, 15), and
  add the anchor M(ω = 0) = 0.
- For each (P, ω) cell, compute the mean over Exp and its jackknife error
  σ_JK = √(N − 1)·std.

**G2. Model selection and the reference curve.**

- Fit eight candidate forms of M(ω, P) to the cell means, for P = 1…5 and ω > 0.
  The fits are weighted least squares with weight 1/σ²_JK and use 30 multi-starts.
- Rank the forms by QAICc = χ²/ĉ + 2k′ (+ a small-sample term). Here ĉ is the
  χ²/dof of the most flexible form and k′ = k + 1. Check the ranking with 200 Exp
  bootstrap resamples.
- The selected form is M(ω, P) = A·ω·e^{−sω}·P^{−γ}, with A, s and γ fitted for
  each FM.
- The fitted curve at P = 3 becomes the reference M₃(ω) on ω = 0, 0.1, …, 20
  (201 points).

**G3. Pair-consistent noise-free curves.**

All P are derived from M₃ so that every momentum pair gives the same I(ω), under
M_P = I + Y/P²:

- Solve M₃ = (1 − C)·M₁ + C·M₂ for M₁ and M₂ by gradient descent. Start halfway
  to the I = 0 point, M_P = 9M₃/(2P²).
- Then I = (9M₃ − M₁)/8 and M_P = I + (M₁ − I)/P² for P = 4, 5.
- The result, Est_M(ω, P) = I(ω) + Y(ω)/P², is the noise-free reference.

**G4. Noisy replicas.**

- For each (FM, P), fit a straight line std(ω) = slope·ω + intercept to the real
  spread over Exp at the measured ω values. Extrapolate it to ω = 20, floored at a
  tiny positive value.
- For each (FM, P, ω), draw N_FM replicas (1026 / 1013 / 900, the same as the
  real Exp count) from Normal(Est_M, std), truncated at M ≥ 0.
- Smooth each replica curve along ω with a Savitzky–Golay filter (window 60
  points, order 3), and set M(0) = 0.
- Result for each FM: (5 P) × N_FM replicas × 201 ω values.

### A3. Supervised learning frames (sliding windows)

For every (FM, P, Exp) synthetic curve on the 0.1 grid, slide a window of 5
consecutive points along ω:

- **Features:** P, the four lag pairs (w₁, m₁), (w₂, m₂), (w₃, m₃), (w₄, m₄),
  and the next ω, w₅. That is 10 features.
- **Target:** m₅ = M(w₅).
- Each curve gives 197 windows.
- **Two feature variants:** "without Exp" (the default, 10 features) and "with
  Exp" (Exp added as an 11th feature). They are trained and stored separately.
- **One frame per FM.** Each FM is its own physical series.

### A4. Models: five regressor types, one model per (FM, P)

For each FM, the frame is split by P and a separate regressor is fit for each P.
So a "model" is a set of 5 per-P estimators per FM.

| Type | Estimator | Tuned hyperparameters (search space) |
| --- | --- | --- |
| GB | gradient boosting | n_estimators 10–300, learning rate 10⁻³–1 (log), max depth 2–10, subsample 0.5–1 |
| RF | random forest (bootstrap) | n_estimators 2–300, max depth 2–50, min samples split 2–20, min samples leaf 1–20 |
| XGB | XGBoost (column subsample 0.8) | n_estimators 10–300, max depth 2–50, η 10⁻³–0.5 (log), subsample 0.5–0.8 |
| NN | MLP with standardised inputs and target; ReLU; layer widths halve (e.g. 64 → 32 → 16); early stopping | width 16–128, depth 1–3, L2 α 10⁻⁶–10⁻¹ (log), learning rate 10⁻⁴–10⁻² (log) |
| GP | Gaussian process with standardised inputs: Constant × ARD-RBF (one length scale per feature) + White noise | kernel tuned by maximum marginal likelihood (2 restarts), on 2000 random rows per P |

### A5. Hyperparameter search and training

- **GB, RF, XGB and NN:** Bayesian optimisation with 20 iterations for each
  (FM, P), scored by mean squared error.
- **Cross-validation is forward in ω, not random.** The distinct w₅ values are cut
  at quantiles into 6 blocks. Each of the 5 folds trains on windows with
  w₅ ≤ cut and validates on the next ω block. This matches how the models are
  used, which is to extrapolate forward in ω.
- **Speed:** the search runs on a random sample of 20 000 rows per P. The winning
  hyperparameters are then refit on the full per-P frame.
- **Seeds:** random seeds are set per P.
- **GP:** no search. The kernel is fit by marginal likelihood on a seeded
  subsample of 2000 rows per P.
- **Training error:** in-sample RMSE and R² for each (FM, P, type, variant).

### A6. Autoregressive extrapolation (inference)

For each FM, P and Exp:

- **Seed:** the first 4 grid points, ω = 0.0, 0.1, 0.2 and 0.3, of that Exp's
  synthetic replica. Real data are never used as seeds, because real and
  synthetic M differ by up to about 2× for some P.
- **Roll forward one grid step (Δω = 0.1) at a time:** the per-P model predicts M
  at the next ω, and that prediction becomes the newest lag. Repeat up to ω = 20.
  There is no teacher forcing.
- **Optional uncertainty modes:**
  - **real-anchored:** each Exp follows its own real M, linearly interpolated,
    across the measured range, then rolls on from there;
  - **GP latent draws:** K extra rolls from the Exp-mean seed. Each step samples
    the GP's latent prediction, correlated step to step, so model uncertainty
    builds up along ω.

### A7. From M to I(ω), and error bars

- For each FM and model, average M over Exp. The error is the jackknife error,
  σ_JK = √((N − 1)/N · Σᵢ(θᵢ − θ̄)²).
- Combine every momentum pair 2 ≤ p₁ < p₂ with the pair relation. Average the
  non-negative pair estimates at each ω (I = 0 where every pair is negative).
- Smooth I(ω) with a Savitzky–Golay filter (window 60, order 3), and set I(0) = 0.

### A8. Test error and model ranking

- **Test error:** compare each model's projected M with the real lattice M at
  every real (P, Exp, ω) point beyond the seed window, which the rollout never
  saw.
  - The projection is linearly interpolated onto the exact real ω values.
  - The comparison is replica by replica (same Exp).
  - Scores: RMSE and R² for each (FM, model).
- **Ranking:** a model's overall score is the average of its rank by mean RMSE
  and its rank by mean R² across FMs, with ties broken by mean RMSE. The #1 model
  is used for the continuum limit.

### A9. Continuum limit a → 0: three routes

**Route 1, pointwise on I.** At each ω, fit the three spacings with
I(ω, a) = I₀(ω) + c(ω)·a² (main); a fit in a is a systematic band.

**Route 2, jackknife on I.**

- The same pointwise fits are applied to I built from each Exp row.
- The fit is linear in the per-FM I and the ensembles are independent, so
  σ²_stat = Σ_FM w_FM²·σ²_JK,FM.
- σ_sys = |I₀(a²) − I₀(a)|, and σ_tot = σ_stat ⊕ σ_sys.

**Route 3, continuum on M first, then I.**

- For each P, fit a global model over every ω > 0 and all three spacings:
  M(ω, a) = M₀(ω) + Σ_k θ_k·g_k(ω, a).
  - M₀(ω) is free at each ω.
  - The cutoff term is shared across ω and vanishes at ω = 0.
- Three cutoff forms, with x = ω/ω_max:
  - O(a): g = a·(x, x²)
  - O(a²): g = a²·(x, x²)
  - O(a/z): g = (a/z)·(x², x³)
- Build I from the continuum M₀ with the same pair rule and smoothing.
- Uncertainty, added in quadrature; bands at 68 % and 95 %:
  - statistical: jackknife over Exp, with each FM in turn;
  - fit: draws from the fit covariance;
  - model: GP latent draws;
  - systematic: the spread across the three cutoff forms.

**Final output:** the continuum I(ω) with error bands.

## Part B: what to draw

### Output specifications

- High resolution: 600 dpi raster at 7.2 in × 6.8 in (4320 × 4080 px). Also
  export a vector PDF or SVG with the identical layout. The figure must stay sharp
  when zoomed and when printed at full journal width.
- White background, flat style: no 3D, no shadows, no gradients, no clip art, no
  logos.
- Font: sans-serif such as Helvetica or Arial. Box titles 9 pt bold, body text
  7–7.5 pt, math in italic serif in LaTeX style with true subscripts and
  superscripts (M₃, P², a², w₅). Every text element must still be readable when
  printed at 7.2 in wide.
- Use a colourblind-safe palette, one colour per stage:
  - synthetic data generation: teal #0f8f9a
  - learning frames: blue #2a78d6
  - models and training: orange #eb6834
  - inference: purple #8e5bd0
  - M → I and evaluation: green #1baf7a
  - continuum: dark grey #555555

  Boxes have rounded corners, a 1 pt outline in the stage colour and a 10 % tint
  of the same colour as fill.
- Arrows: 1 pt, dark grey #333333, with small arrowheads. Label an arrow with what
  it carries where a label is given below.
- Label the generation steps "G1"–"G4" in small teal rounded tags, and every other
  box with a numbered circle ①–⑧, each at its top-left corner.
- Use only the text given here. Do not add result values, a figure title or extra
  commentary.

### Layout: four rows, read left to right and top to bottom

#### Row 0: synthetic data generation (a full-width teal band, about 25 % of the height)

Draw a band titled "Synthetic data generation (per FM)" containing four boxes of
equal width, joined G1 → G2 → G3 → G4.

**G1. Real lattice data.**

- Mini plot: sparse points with error bars for 5 P at a few small ω values.
- Text:
  - "jackknife samples Exp: 1026 / 1013 / 900"
  - "ω = P·z·d₁; keep the 5 / 4 / 3 smallest ω (FM 9 / 12 / 15)"
  - "anchor M(0) = 0"
  - "cell mean ± σ_JK"

**G2. Model selection (QAICc).**

- Mini graphic: a ranked bar list of 8 forms with the top one highlighted.
- Text:
  - "8 forms of M(ω, P), weighted fit 1/σ²_JK, 30 starts"
  - "QAICc = χ²/ĉ + 2k′, 200 bootstrap resamples"
  - "selected: M = A·ω·e^{−sω}·P^{−γ}"
  - "reference M₃(ω): ω = 0…20, Δω = 0.1"

**G3. Pair-consistent curves.**

- Mini plot: 5 smooth M(ω) curves for P = 1…5 and one I(ω) curve beneath them.
- Text:
  - "M₃ = (1 − C)M₁ + C M₂ → M₁, M₂"
  - "I = (9M₃ − M₁)/8"
  - "M_P = I + (M₁ − I)/P², P = 4, 5"
  - "Est_M = I + Y/P²: every pair gives the same I"

**G4. Noisy replicas.**

- Mini plot: an Est_M curve with a band that widens with ω and a few thin replica
  lines.
- Text:
  - "std(ω) = slope·ω + intercept, extrapolated"
  - "N_FM draws ~ Normal(Est_M, std), truncated at M ≥ 0"
  - "Savitzky–Golay (window 60, order 3), M(0) = 0"
  - "(5 P) × N_FM × 201 ω"

Draw a thick arrow from G4 down to ①, labelled "synthetic curves".

#### Row 1: learning frames → models → training

**① Sliding-window frames (blue).**

- Draw one curve of dots along ω with a highlighted window of 5 consecutive dots:
  4 filled dots labelled "lags (w₁,m₁)…(w₄,m₄)" and 1 hollow dot labelled
  "target m₅ at w₅". A small arrow shows the window sliding along ω.
- Below it, a one-row table with the columns:
  "P | w₁ m₁ | w₂ m₂ | w₃ m₃ | w₄ m₄ | w₅ → target".
- Text:
  - "per (FM, P, Exp), Δω = 0.1, 197 windows per curve"
  - "variants: without Exp (10 features) / with Exp (11)"
  - "one frame per FM"

**② Five regressor types, one model per (FM, P) (orange). This is the largest box.**

- Draw five small model cards side by side:
  - "GB · gradient boosting"
  - "RF · random forest"
  - "XGB · XGBoost"
  - "NN · MLP, standardised in/out"
  - "GP · ARD-RBF + white kernel"
- Above the cards, show a split "frame → P = 1 | P = 2 | P = 3 | P = 4 | P = 5"
  with five thin arrows, one per P.
- Text: "a model = 5 per-P estimators per FM".

**③ Hyperparameter search and training (orange).**

- Mini graphic: an ω axis cut into 6 blocks, with 5 fold rows, each training on
  the blocks to the left (filled) and validating on the next block (hatched).
- Text:
  - "Bayesian optimisation, 20 iterations per (FM, P), MSE"
  - "forward-in-ω CV: 5 folds on w₅ (train ≤ cut, validate next block)"
  - "search on 20 000 rows → refit on full per-P frame"
  - "GP: marginal likelihood on 2000 rows, 2 restarts"
  - "training error: RMSE, R²"

#### Row 2: inference → M to I → evaluation

**④ Autoregressive extrapolation (purple).**

- Draw an ω axis from 0 to 20. Show 4 filled seed dots at ω = 0.0–0.3, then a long
  chain of hollow dots, each joined to the previous four by faint arcs, with a
  curved arrow "prediction → newest lag".
- Text:
  - "seed: first 4 grid points (ω = 0.0–0.3) of each Exp's synthetic replica"
  - "step Δω = 0.1, one point at a time, to ω = 20, no teacher forcing"
  - "optional: real-anchored replicas; GP latent draws (model uncertainty)"

**⑤ From M to I(ω) (green).**

- Mini plot: 5 projected M curves with bands, and a funnel arrow to one I(ω)
  curve.
- Text:
  - "Exp mean ± σ_JK, σ_JK² = (N − 1)/N Σ(θᵢ − θ̄)²"
  - "pairs 2 ≤ p₁ < p₂: I = (p₂²M₂ − p₁²M₁)/(p₂² − p₁²)"
  - "average the non-negative pair estimates"
  - "Savitzky–Golay (window 60, order 3), I(0) = 0"

**⑥ Test error and model ranking (green).**

- Mini graphic: real data points (squares) on a projected curve, and a small
  ranked list "#1, #2, #3" of model cards.
- Text:
  - "real points beyond the seed window (never seen by the roll)"
  - "interpolate to the exact real ω, compare replica by replica"
  - "RMSE and R² per (FM, model)"
  - "rank = mean of rank(RMSE) and rank(R²) across FMs"
  - "#1 model → continuum"

#### Row 3: continuum limit → result

**⑦ Continuum extrapolation a → 0 (dark grey). Make it a wide box with three
side-by-side sub-panels.**

- **"Route 1 · pointwise on I":** inset of I against a² with three points at
  a = 0.0888, 0.1207 and 0.151 fm and a line extended to a = 0. Text:
  "I(ω,a) = I₀(ω) + c(ω)a²; fit in a = systematic".
- **"Route 2 · jackknife on I":** text:
  - "per-Exp I, same fits"
  - "σ²_stat = Σ w_FM² σ²_JK,FM"
  - "σ_sys = |I₀(a²) − I₀(a)|"
- **"Route 3 · continuum on M, then I":** text:
  - "M(ω,a) = M₀(ω) + Σ θ_k g_k(ω,a)"
  - "forms: a·(x, x²), a²·(x, x²), (a/z)·(x², x³), x = ω/ω_max"
  - "σ: stat (jackknife) ⊕ fit ⊕ model (GP draws) ⊕ form spread"

**⑧ Result (dark grey, bold outline).**

- A mini plot of a single I(ω) curve with two nested bands, labelled "68 %" and
  "95 %".
- Bold text: "Continuum I(ω) ± σ".

### Connections

- G1 → G2 → G3 → G4 (row 0), then a thick arrow G4 → ① labelled "synthetic curves".
- ① → ② labelled "per-P frames"; ② → ③ (row 1).
- ③ → ④ labelled "trained per-P models"; ④ → ⑤ labelled "projected M(ω, P, Exp)";
  ⑤ → ⑥ (row 2).
- A thin dashed arrow from G1 to ⑥ labelled "real data for the test".
- ⑥ → ⑦ labelled "#1 model"; ⑤ → ⑦ labelled "I(ω) per FM"; ⑦ → ⑧ (row 3).
- A thin dashed arrow from ④ to Route 3, labelled "real-anchored replicas + GP
  draws".

### Style notes

- Align the boxes on a grid, keep even spacing and generous white space, and make
  all boxes in a row the same height.
- Use the same per-P colours (P = 1…5) in every mini plot: blue, orange, green,
  red, purple.
- Keep the mini plots schematic: no tick numbers, only the axis labels ω, M, I and
  a².
- Draw the five model cards in box ② as identical rounded tiles, each with a tiny
  pictogram: a tree stack for GB, a tree cluster for RF, a boosted tree stack with
  a "+" for XGB, a 3-layer network for NN, and a smooth curve with a shaded band
  for GP.
