# Validation: open items

Hyperparameter search now validates forward in omega (`OmegaBlockSplit` in
`model.py`) instead of splitting by Exp replica. Two further issues were
noted and deferred:

## 1. Close the leak between synthetic training data and the real-data test

`test_error.py` scores models on real M past the 4-point seed window, but
those same real points were already used upstream:
`model_selection.py` fits M(omega, P) to *all* real cells ->
`synthetic_data_generator.py` draws the training replicas around that fit.
So the test RMSE is optimistic: it mostly measures how well the models
reproduce the M3 curve.

Options:
- Fit `model_selection.py` / M3 only on the real points used as the seed
  (or a fixed low-omega subset), regenerate synthetic data from that fit,
  and score `test_error.py` only on real points the fit never saw.
- Or hold out a whole ensemble: build synthetic data from FM 9 + 12,
  test on FM 15 (and rotate).

Caveat: real data is sparse (~15-25 cells per FM), so a clean holdout
leaves little to fit the curve on. Decide with the domain scientist.

## 2. Make the real-data test error the headline metric

`train_error.py` is in-sample on the synthetic frames and cannot rank
models (RF / deep XGB look near-perfect by memorising). Report
`test_error.py` (ideally after item 1) as the model-comparison number;
keep `train_error.py` as a sanity check only.
