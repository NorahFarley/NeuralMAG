#!/usr/bin/env bash
set -Eeuo pipefail


# # Run from the directory containing this script and gen_data_new.py.
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

GRID_SIZE=128
MASKED_NUM=200
UNMASKED_NUM=100
SEED_START=0
HEXT='random'

python -u ./gen_data_new.py \
    --w $GRID_SIZE \
    --field-mode $HEXT \
    --seed-start $SEED_START \
    --masked $MASKED_NUM \
    --unmasked $UNMASKED_NUM \
    --output-root ./Dataset/rate_change
