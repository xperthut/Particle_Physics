# DeepQCDConstraint: results

Every number here is from `evaluate.py` or `plot_real_M_I.py`, run with the
checkpoints in `../model/DeepQCDConstraint/`. Each held-out FM is scored with the
model trained on the other two FMs. Errors are RMSE against the noise-free
generated curve `Est_M`. The comparison model is DeepQCD `direct-noa-nolevel`,
trained on the same split.

## 1. The constraint always holds

The violation rate is the fraction of (sample, ω) cells where P²·M_P does not
rise strictly with P.

| Model | Windows | Full rollouts | Real-data rollouts |
| --- | --- | --- | --- |
| DeepQCDConstraint | 0 | 0 | 0 |
| DeepQCD direct-noa-nolevel | up to 0.2% | up to 58% (FM15, Δω = 0.1); about 23% with lattice seeds | not run |

Every momentum pair gives the same I(ω), to floating-point precision, so
I(ω) no longer needs the old clip / abs / signed modes.

## 2. Accuracy on the held-out FM (synthetic data)

**Windows** (4 seeds → 6 targets), RMSE averaged over P:

| FM | Δω | Constraint, grid seeds | DeepQCD, grid seeds | Constraint, lattice seeds |
| --- | --- | --- | --- | --- |
| 9 | 0.1 | 0.0020 | 0.0025 | 0.0006 |
| 9 | 1.0 | 0.0021 | 0.0202 | 0.0085 |
| 12 | 0.1 | 0.0011 | 0.0017 | 0.0012 |
| 12 | 1.0 | 0.0011 | 0.0223 | 0.0034 |
| 15 | 0.1 | 0.0011 | 0.0015 | 0.0004 |
| 15 | 1.0 | 0.0009 | 0.0132 | 0.0009 |

DeepQCD can't take lattice seeds in a window, because its 6 targets for each P
would sit on a different ω grid.

**Full rollouts to ω = 20**, RMSE of the replica-mean M:

- **Grid seeds:** the constrained model is better in all 12 (FM, Δω) cases, by
  2–14×. Examples:
  - FM9, Δω = 0.2: 0.011 against 0.048
  - FM12, Δω = 1.0: 0.003 against 0.045
- **Lattice seeds** (each FM's real ω = P·z·d₁, the case that matches the real
  data): the constrained model is better in 10 of 12 cases. It is worse for
  FM15 at Δω = 0.1 and 0.2: 0.037 and 0.031, against 0.025.
- **I(ω) with lattice seeds:** the constrained model is worse for FM9 at every
  Δω, about 0.008 against 0.0015 (0.004 against 0.0013 at Δω = 1.0). The
  reason is that DeepQCD rolls out each P on its own seed grid.
- **Δω for real data:** the constrained model is most accurate with lattice
  seeds at Δω = 0.5–1.0 for FM12 and FM15, and at Δω = 1.0 for FM9.

Plot: `images/rollout_lattice_dw0.5_FM{9,12,15}.png` (the other settings are scored in `eval_rollout.csv`).

## 3. Real lattice data: the data break the constraint

`plot_real_M_I.py` uses `dqc_all.pt` with Δω = 0.2. On the jackknife means, the
real data break the constraint at 23 of the 24 momentum pairs where two momenta
share an ω (P_i·z_i = P_j·z_j):

- FM9: 12 of 12 pairs
- FM12: 7 of 8 pairs
- FM15: 4 of 4 pairs

For example, at FM9 and ω = 0.785, 1²·M₁ = 0.808 but 4²·M₄ = 0.523. Every
corresponding pair estimate I_ij is negative.

The training data obey the constraint everywhere, so real seeds lie outside the
training distribution. The constrained output then can't be close to them:

- The DNN curves sit 3–10 jackknife errors above the real points (pulls of
  2.7 to 9.8 in `real_vs_dnn_dqc_all_dw0.2.csv`).
- The output I(ω) peaks at about 0.4–0.7, against the generator's 0.05–0.13.

`images/real_M_I_dqc_all_dw0.2_FM{9,12,15}.png` shows this. The model is working as
designed, and the result says the real data and the constraint disagree.

## Open question

Either the real data carry large effects that break the pair constraint, such
as higher-twist terms or lattice artifacts at these z, or the constraint is
meant in another form. Settling which is a physics decision, not a modelling
one.
