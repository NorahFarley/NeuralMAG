#!/usr/bin/env bash

# Run the streamlined NeuralMAG M-H evaluator across the requested parameter sweep.
#
# Expected location:
#   egs/NMI/MH_evaluate/runMH_tensor_gradient.sh
#
# Expected Python files in the same directory:
#   MH_unet_mm.py
#   searcher.py
#   plots.py
#
# Run with:
#   chmod +x runMH_tensor_gradient.sh
#   ./runMH_tensor_gradient.sh

set -Eeuo pipefail

# Resolve paths independently of the directory from which this script is launched.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MH_evaluate -> NMI -> egs -> NeuralMAG repository root
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_SCRIPT="$SCRIPT_DIR/MH_unet_mm.py"

for required_file in "$MAIN_SCRIPT" "$SCRIPT_DIR/searcher.py" "$SCRIPT_DIR/plots.py"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Cannot find $required_file" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Model and shared simulation settings
# ---------------------------------------------------------------------------
GPU=0
KRN=16
LAYERS=2
MODEL_NAME="model.pt"
LOSS_TYPE="baseline_More_6"

MS_BASE=1000
AX_BASE="0.5e-6"
KU_BASE="0.0"
DTIME_SMALL="2.0e-13"
DTIME_LARGE="5.0e-13"
MAX_ITER_SMALL=100000
MAX_ITER_LARGE=200000

SPIN_SPLIT=8
RAND_SEED=1234

HEXT_START=1000
HEXT_END=-1000
HEXT_STEPS=201
FIELD_ANGLE_RADIANS=0.01

# ---------------------------------------------------------------------------
# Plot controls
# ---------------------------------------------------------------------------
# KEEP_ORIGINAL_PLOTS=1 creates one original diagnostic PNG per Hext point.
# For 64 runs x 201 points, that is 12,864 PNGs and substantial plotting I/O.
KEEP_ORIGINAL_PLOTS=0
KEEP_SUMMARY_PLOTS=1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_ROOT="$SCRIPT_DIR/logs_${LOSS_TYPE}"
mkdir -p "$LOG_ROOT"

COMMON_ARGS=(
    --gpu "$GPU"
    --krn "$KRN"
    --layers "$LAYERS"
    --loss_type "$LOSS_TYPE"
    --model_name "$MODEL_NAME"
    --spin_split "$SPIN_SPLIT"
    --rand_seed "$RAND_SEED"
    --hext_start "$HEXT_START"
    --hext_end "$HEXT_END"
    --hext_steps "$HEXT_STEPS"
    --field_angle_radians "$FIELD_ANGLE_RADIANS"
)

if [[ "$KEEP_ORIGINAL_PLOTS" -eq 0 ]]; then
    COMMON_ARGS+=(--skip_original_plots)
fi
if [[ "$KEEP_SUMMARY_PLOTS" -eq 0 ]]; then
    COMMON_ARGS+=(--skip_summary_plots)
fi

run_case() {
    local run_label="$1"
    shift

    local log_file="$LOG_ROOT/${run_label}.log"
    echo
    echo "======================================================================"
    echo "Starting: $run_label"
    echo "Log:      $log_file"
    echo "======================================================================"

    "$PYTHON_BIN" -u "$MAIN_SCRIPT" \
        "${COMMON_ARGS[@]}" \
        "$@" \
        2>&1 | tee "$log_file"
}

# ---------------------------------------------------------------------------
# Widths 64 and 96
# ---------------------------------------------------------------------------
for width in 64 96; do
    # Geometry sweep. "True" invokes the repository's fixed random mask.
    for mask in True triangle hole; do
        run_case \
            "w${width}_mask-${mask}_Ms${MS_BASE}_Ax${AX_BASE}_Ku${KU_BASE}" \
            --w "$width" --Ms "$MS_BASE" --Ax "$AX_BASE" --Ku "$KU_BASE" \
            --dtime "$DTIME_SMALL" --max_iter "$MAX_ITER_SMALL" --mask "$mask"
    done

    # Saturation magnetization sweep on the unmasked square geometry.
    for Ms in 1200 1000 800 600 400; do
        run_case \
            "w${width}_square_Ms${Ms}_Ax${AX_BASE}_Ku${KU_BASE}" \
            --w "$width" --Ms "$Ms" --Ax "$AX_BASE" --Ku "$KU_BASE" \
            --dtime "$DTIME_SMALL" --max_iter "$MAX_ITER_SMALL"
    done

    # Uniaxial anisotropy sweep.
    for Ku in 1e5 2e5 3e5 4e5; do
        run_case \
            "w${width}_square_Ms${MS_BASE}_Ax${AX_BASE}_Ku${Ku}" \
            --w "$width" --Ms "$MS_BASE" --Ax "$AX_BASE" --Ku "$Ku" \
            --Kvec 1,0,0 --dtime "$DTIME_SMALL" --max_iter "$MAX_ITER_SMALL"
    done

    # Exchange-stiffness sweep.
    for Ax in 0.7e-6 0.6e-6 0.4e-6 0.3e-6; do
        run_case \
            "w${width}_square_Ms${MS_BASE}_Ax${Ax}_Ku${KU_BASE}" \
            --w "$width" --Ms "$MS_BASE" --Ax "$Ax" --Ku "$KU_BASE" \
            --dtime "$DTIME_SMALL" --max_iter "$MAX_ITER_SMALL"
    done
done

# ---------------------------------------------------------------------------
# Widths 128 and 256
# ---------------------------------------------------------------------------
for width in 128 256; do
    for Ms in 1200 1000 800 600 400; do
        run_case \
            "w${width}_square_Ms${Ms}_Ax${AX_BASE}_Ku${KU_BASE}" \
            --w "$width" --Ms "$Ms" --Ax "$AX_BASE" --Ku "$KU_BASE" \
            --dtime "$DTIME_LARGE" --max_iter "$MAX_ITER_LARGE"
    done

    for Ku in 1e5 2e5 3e5 4e5; do
        run_case \
            "w${width}_square_Ms${MS_BASE}_Ax${AX_BASE}_Ku${Ku}" \
            --w "$width" --Ms "$MS_BASE" --Ax "$AX_BASE" --Ku "$Ku" \
            --Kvec 1,0,0 --dtime "$DTIME_LARGE" --max_iter "$MAX_ITER_LARGE"
    done

    for Ax in 0.7e-6 0.6e-6 0.4e-6 0.3e-6; do
        run_case \
            "w${width}_square_Ms${MS_BASE}_Ax${Ax}_Ku${KU_BASE}" \
            --w "$width" --Ms "$MS_BASE" --Ax "$Ax" --Ku "$KU_BASE" \
            --dtime "$DTIME_LARGE" --max_iter "$MAX_ITER_LARGE"
    done

    for mask in True triangle hole; do
        run_case \
            "w${width}_mask-${mask}_Ms${MS_BASE}_Ax${AX_BASE}_Ku${KU_BASE}" \
            --w "$width" --Ms "$MS_BASE" --Ax "$AX_BASE" --Ku "$KU_BASE" \
            --dtime "$DTIME_LARGE" --max_iter "$MAX_ITER_LARGE" --mask "$mask"
    done
done

echo
echo "All runs completed."
echo "Logs: $LOG_ROOT"
