#!/bin/bash

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# Define common parameters
gpu=0
max_iter=50000

# ---------------------------------------------------------
# 1. GENERATE TRAINING DATASETS (Sizes 32, 64, 96)
# Total 300 seeds per size: 2/3 Masked (200), 1/3 Unmasked (100)
# ---------------------------------------------------------
for w in 32 64 96; do
    echo "=========================================="
    echo "Generating TRAINING data for Grid Size: ${w}x${w}"
    echo "=========================================="

    # Part A: 2/3 Masked Seeds (200 Seeds) with randomized Hext (scaled to 1000 Oe max)
    python -m utils.gen_data \
        --w $w \
        --Hext_val 1000 \
        --nseeds 200 \
        --mask 'True' \
        --gpu $gpu \
        --max_iter $max_iter

    # Part B: 1/3 Unmasked Seeds (100 Seeds) with randomized Hext
    python -m utils.gen_data \
        --w $w \
        --Hext_val 1000 \
        --nseeds 100 \
        --mask 'False' \
        --gpu $gpu \
        --max_iter $max_iter
done

# ---------------------------------------------------------
# 2. GENERATE CROSS-SCALE TEST DATASET (Size 128)
# 100 seeds, completely unmasked, zero external field (Hext = 0)
# ---------------------------------------------------------
echo "=========================================="
echo "Generating CROSS-SCALE TEST data for Grid Size: 128x128"
echo "=========================================="

python -m utils.gen_data \
    --w 128 \
    --Hext_val 0 \
    --nseeds 100 \
    --mask 'False' \
    --gpu $gpu \
    --max_iter $max_iter

echo "All Dataset Generation Pipelines Completed Successfully!"