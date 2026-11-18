#!/usr/bin/env bash
# Fresh, non-smoothed run of the Helicity 2026 pipeline into new folders:
#   Data/<NAME>   (synthetic data, train frames, projections, test error, Images/)
#   model/<NAME>  (trained .sav models)
# Nothing in Data/2026, model/2026 or the *_nosmooth backups is read or written,
# except the raw inputs rpitd/ and ITD-pol-fit1-line.txt, which are copied from Data/2026.
#
# synthetic_data_generator_abg.py and prepare_train_test_data.py hard-code
# ../Data/2026, so this script runs temporary patched copies of those two
# (deleted on exit); the original .py files are not modified.
#
# Usage (from anywhere):
#   bash Helicity/Code/run_fresh_nosmooth.sh
# Options via environment variables:
#   NAME=2026_fresh                 output folder name under Data/ and model/
#   PY=~/miniconda3/envs/qsar/bin/python
#   MODELS="GB XGB RF NN GP BNN"    model types to train
#   EXP_IDS="No Yes"                variants to train/reconstruct ("No" only = faster)
#   TEST_MODEL=RF                   model for best_model_I_* in test_error.py
# e.g.  NAME=2026_fresh2 EXP_IDS="No" MODELS="RF NN" bash run_fresh_nosmooth.sh

set -euo pipefail

NAME="${NAME:-2026_fresh}"
PY="${PY:-$HOME/miniconda3/envs/qsar/bin/python}"
MODELS="${MODELS:-GB XGB RF NN GP BNN}"
EXP_IDS="${EXP_IDS:-No Yes}"
TEST_MODEL="${TEST_MODEL:-RF}"

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"

D="../Data/$NAME"
M="../model/$NAME"
SRC="../Data/2026"
LOG_DIR="$D/logs"

if [[ "$NAME" == "2026" || "$NAME" == "2026_nosmooth" ]]; then
    echo "NAME=$NAME would overwrite existing data; pick another name." >&2
    exit 1
fi
if [[ -e "$D/All_p_w_m.csv" || -d "$M" ]]; then
    echo "$D or $M already has results; remove them or set NAME=... to another folder." >&2
    exit 1
fi
[[ -x "$PY" ]] || { echo "python not found: $PY (set PY=...)" >&2; exit 1; }

# --- 1. raw inputs ---------------------------------------------------------
mkdir -p "$D" "$M" "$LOG_DIR"
cp -R "$SRC/rpitd" "$D/"
cp "$SRC/ITD-pol-fit1-line.txt" "$D/"

# --- temporary copies of the two scripts with the hard-coded folder patched -
GEN="_fresh_synthetic_data_generator_abg.py"
PREP="_fresh_prepare_train_test_data.py"
trap 'rm -f "$CODE_DIR/$GEN" "$CODE_DIR/$PREP"' EXIT

sed "s|^VERSION = 2026\$|VERSION = '$NAME'|" synthetic_data_generator_abg.py > "$GEN"
sed "s|^    DATA_DIR = '../Data/2026'\$|    DATA_DIR = '$D'|" prepare_train_test_data.py > "$PREP"
grep -q "^VERSION = '$NAME'\$" "$GEN"   || { echo "patching $GEN failed" >&2; exit 1; }
grep -q "DATA_DIR = '$D'\$" "$PREP"     || { echo "patching $PREP failed" >&2; exit 1; }

run() {   # run <log name> <command...>: echo, run, tee into $LOG_DIR/<log name>.log
    local name="$1"; shift
    echo
    echo "=== [$(date '+%F %T')] $name: $*"
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

# --- 2. synthetic data (no per-replica smoothing) -------------------------
run generate "$PY" "$GEN" --no_smooth

# --- 3. sliding-window train frames ---------------------------------------
run prepare "$PY" "$PREP"

# --- 4. train --------------------------------------------------------------
for exp in $EXP_IDS; do
    for t in $MODELS; do
        run "train_${t}_exp_${exp}" "$PY" train.py --fm all --with_exp_id "$exp" \
            --model_type "$t" --data_dir "$D" --model_dir "$M"
    done
done

# --- 5. training error -> $D/train_error_summary.csv -----------------------
run train_error "$PY" train_error.py --data_dir "$D" --model_dir "$M"

# --- 6. reconstruct (synthetic seed) ---------------------------------------
for exp in $EXP_IDS; do
    run "reconstruct_exp_${exp}" "$PY" reconstruct_real_data.py --exp_id "$exp" \
        --data_dir "$D" --model_dir "$M" --models $MODELS
done

# --- 7. test error, ranking, plots -----------------------------------------
if [[ " $EXP_IDS " == *" Yes "* && " $EXP_IDS " == *" No "* ]]; then
    VARIANT=both
elif [[ " $EXP_IDS " == *" Yes "* ]]; then
    VARIANT=with_exp
else
    VARIANT=no_exp
fi
run test_error "$PY" test_error.py --model "$TEST_MODEL" --variant "$VARIANT" --data_dir "$D" --ylim_I -0.02 0.1

echo
echo "Done [$(date '+%F %T')]. Data: Helicity/Data/$NAME   models: Helicity/model/$NAME   logs: $LOG_DIR"
