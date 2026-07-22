#!/bin/bash

set -euo pipefail

# Absolute path to the NeuralMAG repository.
PROJECT_ROOT="/lustre/home/farleyn/NeuralMAG"
PYTHON_SCRIPT="${PROJECT_ROOT}/utils/gen_data_new.py"

cd "${PROJECT_ROOT}"

GRID_SIZE=128
MASKED_NUM=200
UNMASKED_NUM=100
SEED_START=0
HEXT='random'

python "${PYTHON_SCRIPT}" \
    --w $GRID_SIZE \
    --field-mode $HEXT \
    --seed-start $SEED_START \
    --masked $MASKED_NUM \
    --unmasked $UNMASKED_NUM \
    --output-root ./Dataset/rate_change
