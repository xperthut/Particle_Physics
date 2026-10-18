# DeepQCD: results

**Question:** suppose a new lattice ensemble at a different lattice spacing *a*
gives M only at small ω. Can one network, trained on the other ensembles,
extrapolate its M(ω) to large ω?

**Test:** leave one FM out. Each model trains on the generated replicas of two
FMs and is scored on the third, which it has never seen. FM12 held out is
interpolation in *a*. FM9 held out (the finest a = 0.0888 fm) is extrapolation,
which is the realistic future case.

All numbers are RMSE against the noise-free generated curve `Est_M`.
Everything is computed on generated data only.

## 1. With all inputs, the network does not generalise

The first models received a², P, the window's ω values, the 4 seed values
relative to the last seed, and the absolute level log(last seed).

- Within the training ensembles they fit well: validation loss on held-out
  replicas was about 0.004–0.008.
- On the unseen FM they failed. Window RMSE was 0.05–0.17, and even the m3 head,
  which has the exact A·ω·e^(−sω) form, was 10–20× worse than a plain
  least-squares fit of that form to the same 4 seeds.

**Diagnosis:** with only two training values of *a*, the network treats a², and
the absolute level of M (peak M differs about 2.7× between FM9 and FM15), as an
ensemble label. It recalls that ensemble's curve instead of reading the shape
from the seeds.

## 2. Ablation: removing a² and the level fixes generalisation

`-noa` removes the a² input; `-nolevel` also removes log(last seed). The network
then sees only the shape of the seeds.

Window test: predict 6 points after 4 seeds.

| Held-out FM, stride | `direct` | `direct-noa` | `m3-noa` | **`direct-noa-nolevel`** | M3 seed fit | Lag XGB (other FM) |
| --- | --- | --- | --- | --- | --- | --- |
| 9, Δω=0.1 | 0.110 | 0.126 | 0.031 | **0.0038** | 0.0043 | 0.149 / 0.189 |
| 9, Δω=1.0 | 0.172 | 0.161 | 0.081 | 0.026 | **0.0062** | n/a |
| 12, Δω=0.1 | 0.0084 | 0.0091 | 0.0037 | **0.0025** | 0.0029 | 0.042 / 0.026 |
| 12, Δω=1.0 | 0.065 | 0.043 | 0.037 | 0.029 | **0.0034** | n/a |
| 15, Δω=0.1 | 0.053 | 0.0038 | 0.0037 | **0.0023** | 0.0023 | 0.025 / 0.0099 |
| 15, Δω=1.0 | 0.081 | 0.035 | 0.034 | 0.017 | **0.0033** | n/a |

- At small strides (Δω = 0.1, 0.2), `direct-noa-nolevel` matches or slightly
  beats the M3 seed fit on all three held-out FMs, including FM9.
- At wide strides, the M3 fit is 4–8× better. On this data the M3 form is
  exactly right, so a fit of that form is hard to beat.
- The network beats the existing lag XGB models by 10–50× when those are applied
  to an ensemble they weren't trained on.
- `m3-noa-nolevel` broke. The m3 head has to output log A, which depends on the
  absolute scale, and that input was removed. This is a design flaw in that
  combination, not a finding.

## 3. Full extrapolation to ω = 20 from 4 seeds

Each replica is seeded at ω = 0, Δω, 2Δω, 3Δω and rolled out to ω = 20. The
table shows the RMSE of the mean curve over the extrapolated points, averaged
over P.

| Held-out FM | Δω | `direct` | **`direct-noa-nolevel`** | M3 seed fit |
| --- | --- | --- | --- | --- |
| 9 | 0.1 | 0.173 | **0.139** | 0.202 |
| 9 | 0.2 | 0.165 | **0.038** | 0.089 |
| 9 | 0.5 | 0.158 | 0.030 | **0.0044** |
| 9 | 1.0 | 0.138 | 0.033 | **0.016** |
| 12 | 0.1 | 0.067 | **0.042** | 0.099 |
| 12 | 0.2 | 0.066 | **0.034** | 0.050 |
| 12 | 0.5 | 0.065 | 0.044 | **0.0027** |
| 15 | 0.1 | 0.077 | **0.029** | 5.26 (unstable) |
| 15 | 0.5 | 0.072 | 0.017 | **0.0013** |

- **Seeds bunched at small ω** (Δω ≤ 0.2, all seeds within ω ≤ 0.6): the
  network is the most reliable method. The M3 fit becomes unstable because its 2
  parameters are poorly constrained, giving RMSE 5.3 for FM15 and jackknife
  errors up to about 970.
- **Seeds spread over ω ≈ 0–3:** the M3 fit is better. The network tracks the
  peak but its tail drifts high (`Images/extrapolated_M_FM9_stride5.png`).
- **Error bars:** at FM12 and FM15 with Δω ≥ 0.2, the network's jackknife bands
  are 2–14× narrower than the M3 fit's (6× and 14× at Δω = 0.5). The network's
  band is the wider one at FM12 with Δω = 0.1 and at FM9 with Δω = 1.0.

## Conclusions

1. **The *a* input hurts.** With 3 ensembles, a network should not receive a² or
   the absolute M level. Any *a*-dependence it needs comes through the seeds
   themselves. Revisit this once there are enough ensembles to learn a real
   dependence on *a*.
2. **For a new ensemble, the method depends on how far the seeds reach:**
   - If the small-ω data reaches ω ≈ 2–3 and M3 is the right form, a direct M3
     fit is the most accurate.
   - If the data covers only a very short ω range, or you don't want to commit
     to the M3 form, use the `direct-noa-nolevel` network. It is stable where the
     fit is not, and it carries over from other ensembles without retraining.
3. **Caveat:** every target here is generated from the M3 model, so the M3 fit is
   effectively evaluated on its own model. On real lattice data that departs
   from M3, the network's lack of an assumed form could matter more. That needs
   real data to test.

## Final model

`../model/DeepQCD/deepqcd_direct-noa-nolevel_all.pt` is trained on all three FMs
with the same settings. It is the model to use for a new ensemble. Because a² is
dropped, it needs only P, the ω values and the seed M values.

## Reproduce

See `README.md`. The six designs (`direct`, `m3`, and each with `--drop a` and
`--drop a,level`) are trained for each held-out FM, then:

```bash
python evaluate.py      # eval_*.csv, Images/eval_step_FM*.png
python extrapolate.py   # extrapolated_M_*.csv/png, extrapolation_summary.csv
```
