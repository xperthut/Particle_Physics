# DeepQCDConstraint

A variant of `../DeepQCD` whose output always satisfies the momentum-pair
constraint. For momenta i < j at the same ω:

  Mᵢ / Mⱼ < j² / i²  ⇔  i²·Mᵢ < j²·Mⱼ  ⇔  every pair estimate Iᵢⱼ = (j²Mⱼ − i²Mᵢ)/(j² − i²) > 0

DeepQCD predicts each P on its own, so it can't enforce a relation between
different P. This network predicts all five P together.

- **Output head:** at each target ω the network outputs log(M₁/s) and log(I/s),
  where s is the largest seed M in the window. Then:

  M_P = I + (M₁ − I)/P²,  i.e.  P²·M_P = M₁ + (P² − 1)·I

  Because I > 0 and M₁ > 0, three things hold by construction:
  - P²·M_P rises strictly with P, which is the constraint for every pair;
  - M_P > 0 for every P;
  - every momentum pair gives exactly the same I(ω).

  This is the generator's own form, M_P = I + Y/P² with Y = M₁ − I
  (`Code/synthetic_data_generator.py`), so the noise-free training curves are
  exactly representable.
- **Input (46 values):** each P's 4 seed ω values and 4 seed M values (M divided
  by s), plus the 6 target ω values. There's no a² or absolute-level input, for
  the reason given in `../DeepQCD/RESULTS.md`: with those inputs the network
  doesn't generalise to an unseen FM.
- **Seed layouts:** training mixes two layouts 50/50. A rollout uses one of them
  for its first block; every later block is a grid block.
  - `grid`: every P is seeded at the same ω values, ω₀ + k·Δω.
  - `lattice`: each P is seeded at its own real lattice ω values, P·z·d₁ for
    z = 0..3, with d₁ between 0.15 and 0.45. The real data have d₁ = 0.196,
    0.262 and 0.393 for FM 9, 12 and 15. The targets are a common grid,
    Δω·(1..6).
- **Loss:** MSE of (M_pred − M_replica) divided by each P's largest seed, so
  P = 5 counts as much as P = 1. Replica `Exp` = e of every P is taken as one
  joint sample.
- **Validation:** each model trains on two FMs and is tested on the third, as
  in DeepQCD. Within the training FMs, 10% of the replicas are held back for
  early stopping.

## Steps

Run everything from `Helicity/DeepQCDConstraint` with the `gluon` env:

```bash
for fm in 9 12 15; do python train.py --test_fm $fm; done
python evaluate.py                  # held-out-FM scores, next to DeepQCD direct-noa-nolevel
python train.py --test_fm none      # final model on all FMs
python plot_real_M_I.py --dw 0.2    # real lattice seeds -> M(omega), I(omega)
python continuum_I.py               # a -> 0 of I(omega), real and rollout (Code/continuum_I_jackknife.py method)
python plot.py                      # redraw every image from the CSVs alone (no model needed)
```

| Script | Writes |
| --- | --- |
| `train.py` | `../model/DeepQCDConstraint/dqc_test_FM{fm}.pt` (`dqc_all.pt`), `train_history_*.csv`, `train_log_*.txt`, `images/train_loss_*.png` |
| `evaluate.py` | `eval_windows.csv`, `eval_rollout.csv`, `rollout_{M_all_P,seeds,I_omega}_lattice_dw0.5.csv` + their images (pick others with `--plots`) |
| `plot_real_M_I.py` | `real_M_all_P_*.csv`, `real_I_omega_*.csv`, `real_vs_dnn_*.csv`, `images/real_M_I_*_FM{fm}.png` + `real_I_all_FM_*.png` |
| `plot.py` | every image in `images/`, from the CSVs; `--only train,rollout,real,continuum` |
| `continuum_I.py` | `I_omega_continuum_{real,rollout}.csv`, `images/I_omega_continuum_{real,rollout}.png` |

All data files go to `../Data/DeepQCDConstraint/` and images to
`../Data/DeepQCDConstraint/images/`. `jackknife.py` is copied unchanged from
`../DeepQCD/`. Results are in `RESULTS.md`.
