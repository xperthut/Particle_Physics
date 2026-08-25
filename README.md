# Particle Physics: ML Reconstruction of Gluon PDFs

Generative machine learning code for reconstructing **polarized (helicity)**
and **unpolarized gluon parton distribution functions (PDFs)** from lattice
QCD matrix elements computed at short distance and large momentum.

Real lattice data is sparse in the Ioffe-time variable `W`, so the pipeline
fits a small set of gradient-boosted, random forest, and XGBoost regressors
to synthetically generated pseudo-experiment replicas, then uses those models
to autoregressively extrapolate/reconstruct the matrix elements at values of
`W` beyond what was directly simulated. The reconstructed matrix elements are
then converted into Ioffe-time distributions (ITDs) and compared against
existing PDF fits (e.g. NNPDFpol).

## Repository layout

- **`Helicity/`** — polarized (helicity) gluon PDF pipeline. The more
  complete pipeline, implemented as standalone scripts in `Code/`
  (`generator.py`, `prepare_train_test_data.py`, `model.py`, `train.py`,
  `train_error.py`, `reconstruct_real_data.py`), with supporting notebooks
  for exploration/plotting. `HelicityITDplots/` contains the scripts that
  produce the final ITD plots.
- **`Unpolarized/`** — unpolarized gluon PDF pipeline. Same conceptual
  stages (preprocessing → training/testing → error analysis), implemented
  directly in the notebooks under `Code/`.

Each pipeline is self-contained (own `Data/` and model-output folder) and
scripts use relative paths, so they must be run from their own `Code/`
directory.

## Running the pipeline (Helicity)

From `Helicity/Code/`, in order:

```bash
# 1. Generate synthetic training data from raw lattice files
python generator.py

# 2. Build sliding-window train/test frames
python prepare_train_test_data.py

# 3. Train a model (model_type: GB | RF | XGB; with_exp_id: Yes | No)
python train.py --train_data '../Data/Synthetic_train_data_without_exp.csv' --with_exp_id 'No' --model_type 'GB'

# 4. Check training error (RMSE) of saved models
python train_error.py

# 5. Reconstruct/extrapolate the real data using the trained models
python reconstruct_real_data.py
```

See `Helicity/Code/command.txt` for the full matrix of training invocations.
Post-processing and ITD plot generation live in
`Helicity/Code/post_processing.ipynb` and `Helicity/HelicityITDplots/`.

The `Unpolarized/` pipeline follows the same stages via
`Unpolarized/Code/preprocessing.ipynb`, `training_testing.ipynb`, and
`training_error.ipynb`.

## Citation

If you use this code, please cite:

```bibtex
@article{chowdhury2025polarized,
  title={Polarized and unpolarized gluon PDFs: Generative machine learning applications for lattice QCD matrix elements at short distance and large momentum},
  author={Chowdhury, Talal Ahmed and Izubuchi, Taku and Kamruzzaman, Methun and Karthik, Nikhil and Khan, Tanjib and Liu, Tianbo and Paul, Arpon and Schoenleber, Jakob and Sufian, Raza Sabbir},
  journal={Physical Review D},
  volume={111},
  number={7},
  pages={074509},
  year={2025},
  publisher={APS}
}
```
