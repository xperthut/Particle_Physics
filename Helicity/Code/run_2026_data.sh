#!/usr/bin/env bash
# Regenerate every Data/2026 data file and image from the raw inputs, using the models
# already saved in model/2026 (NOTHING is trained).
#
# Kept inputs (must exist): Data/2026/All_p_w_m.csv, ITD-pol-fit1-line.txt, rpitd/
#
# The model/2026 models were trained on the NON-smoothed synthetic replicas (Nov 5-11,
# before per-replica smoothing became the generator default), so the replicas are drawn
# with --no_smooth by default.  SMOOTH=1 draws smoothed replicas instead (models then
# no longer match their training data).
#
# Usage (from anywhere):
#   bash Helicity/Code/run_2026_data.sh             # all steps
#   FROM=4 bash Helicity/Code/run_2026_data.sh      # resume at step 4
#   SMOOTH=1 bash Helicity/Code/run_2026_data.sh
# Log: Data/2026/run_2026_data.log
set -euo pipefail
PY="${PY:-$HOME/miniconda3/envs/qsar/bin/python}"   # sklearn 1.8 + torch (models were pickled there)
cd "$(dirname "${BASH_SOURCE[0]}")"
DATA=../Data/2026
FROM="${FROM:-1}"
SMOOTH_FLAG=$([[ "${SMOOTH:-0}" == "1" ]] && echo "" || echo "--no_smooth")
LOG=$DATA/run_2026_data.log

for f in All_p_w_m.csv ITD-pol-fit1-line.txt rpitd; do
  [[ -e $DATA/$f ]] || { echo "missing input $DATA/$f"; exit 1; }
done
mkdir -p $DATA/Images
exec > >(tee -a "$LOG") 2>&1

step() { local n=$1; shift; (( n >= FROM )) || return 0
         echo; echo "=== [$(date '+%F %T')] step $n: $*"; "$PY" -u "$@"; }

# 1. synthetic M (alpha/beta/gamma fit, new_M.csv, new_M_S.csv), real-data diagnostics,
#    Exp replicas -> generated_Synthetic_exp_data.csv  (+ diagnostic images)
step 1 synthetic_data_generator_abg.py $SMOOTH_FLAG
# 2. per-FM sliding-window frames -> generated_Synthetic_exp_data_{fm}.csv,
#    Synthetic_train_data_{with,without}_exp_{fm}.csv
step 2 prepare_train_test_data.py
# 3. training error of the saved models -> train_error_summary.csv
step 3 train_error.py
# 4. roll the saved models from the synthetic seed -> Real_data_projected_{fm}_{no,with}_exp_v2.csv,
#    plus the real-anchored roll + GP draws used by continuum_M_extrapolation.py
step 4 reconstruct_real_data.py --exp_id No
step 4 reconstruct_real_data.py --exp_id Yes
step 4 reconstruct_real_data.py --anchor_real --n_draws 200 --models GP
# 5. M -> I, test error, ranking, projected_I / best_model_I figures (RF central)
step 5 test_error.py --model RF --variant both --selectmodel RF NN
# 6. all-FM I(omega): overall #1, RF and NN
step 6 combine_I_omega.py
step 6 combine_I_omega.py --model RF
step 6 combine_I_omega.py --model NN
# 7. continuum a -> 0: pointwise, jackknife (RF central, |RF-NN| model sys kept separate), on M
#    (--model given explicitly: the ranking #1 is NN, but step 5 writes best_model_I only for RF,
#     and the anchored roll of step 4 has only M_GP)
step 7 continuum_extrapolation.py --model RF
step 7 continuum_I_jackknife.py --source roll --model RF --sys_models NN
step 7 continuum_M_extrapolation.py --source anchored --n_draws 500 --model GP
# 8. every remaining figure plot_all.py knows (skips groups whose inputs do not exist)
step 8 plot_all.py

# check: the regenerated synthetic data should match what the models were trained on
if (( FROM <= 2 )) && [[ -z "${SMOOTH:-}" || "${SMOOTH}" == "0" ]] && [[ -d ../Data/2026_nosmooth ]]; then
  echo; echo "=== check vs Data/2026_nosmooth (models' training data)"
  for f in new_M.csv new_M_S.csv generated_Synthetic_exp_data_9.csv \
           Synthetic_train_data_without_exp_9.csv Synthetic_train_data_with_exp_9.csv; do
    if cmp -s $DATA/$f ../Data/2026_nosmooth/$f; then echo "  identical  $f"
    else echo "  DIFFERS    $f"; fi
  done
fi
echo; echo "Done [$(date '+%F %T')]"
