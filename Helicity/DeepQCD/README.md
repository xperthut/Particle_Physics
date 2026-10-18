# DeepQCD

A PyTorch window network that extrapolates M(ω) to large ω from 4 small-ω seed
points, using one model for every lattice spacing *a* and momentum P. It is
trained on the generated replicas only
(`../Data/2026/generated_Synthetic_exp_data.csv`, read as-is and cached as
`../Data/DeepQCD/curves.npz`).

- **Input:** a² (a = 0.0888 / 0.1207 / 0.151 fm for FM 9 / 12 / 15), P, the 10 ω
  values of the window, and M at the first 4 of them.
- **Output:** M at the remaining 6.
- **Strides:** windows are spaced by 1, 2, 5 or 10 grid steps (Δω = 0.1 / 0.2 /
  0.5 / 1.0).
- **Heads:**
  - `direct` predicts the 6 values freely.
  - `m3` predicts the M3 parameters (log A, s) of M = A·ω·e^(−sω) for the window
    and evaluates that curve.
- **`--drop a,level`:** feeds the a² input and log(last seed) as 0. This is the
  design that generalises to an unseen FM; see `RESULTS.md`.
- **Validation:** each model trains on two FMs and is tested on the third. Within
  the training FMs, 10% of the replicas are held back for early stopping.

## Setup

```bash
conda env create -f ../../environment.yml     # once; creates the "gluon" env
conda activate gluon
cd Helicity/DeepQCD                          # all paths are relative to here
```

## Steps

```bash
for fm in 9 12 15; do
  python train.py --test_fm $fm --head direct --drop a,level   # best design (RESULTS.md)
  # ablations: --head direct|m3 with --drop '' | a | a,level
done
python evaluate.py                           # window scores, every checkpoint per held-out FM
python extrapolate.py                        # full curves to ω = 20 from 4 seeds
python train.py --test_fm none --head direct --drop a,level   # final model on all FMs
```

| Script | Writes |
| --- | --- |
| `train.py` | `../model/DeepQCD/deepqcd_{head}_test_FM{fm}.pt`, `../Data/DeepQCD/train_history_*.csv`, `Images/train_loss_*.png` |
| `evaluate.py` | `eval_by_step.csv`, `eval_by_P.csv`, `eval_summary.csv`, `Images/eval_step_FM{fm}.png` |
| `extrapolate.py` | `extrapolated_M_FM{fm}_stride{s}.csv`, `extrapolation_summary.csv`, `Images/extrapolated_M_*.png` (`--use_all` for the all-FM models) |

Baselines:

- **`M3_seed`:** fits A·ω·e^(−sω) to the 4 seeds.
- **`Lag_XGB_FM{y}`:** the existing 4-lag XGB models in `../model/2026/`, stride 1
  only. The GB models were pickled with scikit-learn 1.3 and don't load in this
  environment, so they are skipped.

`jackknife.py` is copied unchanged from `../Code/`. Results are in `RESULTS.md`.
