#!/usr/bin/env bash

# Run the integrated NeuralMAG M-H evaluation pipeline (Parts 1-5).
#
# Expected location:
#   egs/NMI/MH_evaluate/runMH_integrated_parts1_to_5.sh
#
# Expected Python files in the same directory:
#   MH_unet_mm.py
#   searcher.py
#   plots.py
#
# Run with:
#   chmod +x runMH_integrated_parts1_to_5.sh
#   ./runMH_integrated_parts1_to_5.sh

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Resolve paths robustly, independent of the directory from which this script
# is launched.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MH_evaluate -> NMI -> egs -> NeuralMAG repository root
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_SCRIPT="$SCRIPT_DIR/MH_unet_mm.py"

if [[ ! -f "$MAIN_SCRIPT" ]]; then
    echo "ERROR: Cannot find $MAIN_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT_DIR/searcher.py" ]]; then
    echo "ERROR: Cannot find $SCRIPT_DIR/searcher.py" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT_DIR/plots.py" ]]; then
    echo "ERROR: Cannot find $SCRIPT_DIR/plots.py" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Model and shared simulation settings
# ---------------------------------------------------------------------------
GPU=0
KRN=16
LAYERS=2
MODEL_NAME="model.pt"
LOSS_TYPE="baseline_full"

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
# Analysis settings
# ---------------------------------------------------------------------------
# Set FINAL_STATISTICS=1 for manuscript-quality resampling statistics.
# Leave it at 0 for faster screening/debugging runs.
FINAL_STATISTICS=1
if [[ "$FINAL_STATISTICS" -eq 1 ]]; then
    INDICATOR_PERMUTATIONS=5000
    INDICATOR_BOOTSTRAP=2000
else
    INDICATOR_PERMUTATIONS=500
    INDICATOR_BOOTSTRAP=500
fi

INDICATOR_PRIMARY_WINDOW=10
INDICATOR_MAX_LAG=20
INDICATOR_TOP_N=12

PUBLICATION_TOP_N=4
PUBLICATION_PRE_STEPS=20
PUBLICATION_POST_STEPS=10
PUBLICATION_DPI=300
PUBLICATION_FORMATS="png,pdf"

MANUSCRIPT_TOP_N=10
MANUSCRIPT_FORMATS="csv,tex,md"

# ---------------------------------------------------------------------------
# Plot controls
# ---------------------------------------------------------------------------
# KEEP_ORIGINAL_PLOTS=1 preserves the original repository plot_results()
# figure at every Hext step. Across this full sweep that creates 12,864 PNGs
# (64 runs x 201 field points), so storage and plotting overhead are large.
KEEP_ORIGINAL_PLOTS=1
KEEP_SUMMARY_PLOTS=1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_ROOT="$SCRIPT_DIR/logs_integrated_${LOSS_TYPE}"
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
    --indicator_primary_window "$INDICATOR_PRIMARY_WINDOW"
    --indicator_max_lag "$INDICATOR_MAX_LAG"
    --indicator_permutations "$INDICATOR_PERMUTATIONS"
    --indicator_bootstrap "$INDICATOR_BOOTSTRAP"
    --indicator_top_n "$INDICATOR_TOP_N"
    --publication_top_n "$PUBLICATION_TOP_N"
    --publication_pre_steps "$PUBLICATION_PRE_STEPS"
    --publication_post_steps "$PUBLICATION_POST_STEPS"
    --publication_dpi "$PUBLICATION_DPI"
    --publication_formats "$PUBLICATION_FORMATS"
    --manuscript_top_n "$MANUSCRIPT_TOP_N"
    --manuscript_formats "$MANUSCRIPT_FORMATS"
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
        --run_label "$run_label" \
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

    # Saturation magnetization sweep on the default unmasked square geometry.
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

# ---------------------------------------------------------------------------
# Aggregate all completed runs once, after the full sweep.
# This calls CrossSweepAggregator directly so it does not launch an extra M-H
# simulation merely to aggregate existing results.
# ---------------------------------------------------------------------------
AGGREGATE_ROOT="$SCRIPT_DIR/figs_k${KRN}/model_${LOSS_TYPE}"
AGGREGATE_OUTPUT="$AGGREGATE_ROOT/cross_sweep_aggregate"

"$PYTHON_BIN" - <<PY
from pathlib import Path
from egs.NMI.MH_evaluate.searcher import CrossSweepAggregator

root = Path(r"$AGGREGATE_ROOT")
out = Path(r"$AGGREGATE_OUTPUT")
summary = CrossSweepAggregator(root, min_runs=2).run(out)
print("Cross-sweep aggregation complete:")
print(summary)
PY

echo
echo "All runs completed."
echo "Aggregate results: $AGGREGATE_OUTPUT"
