#!/bin/bash

export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH

GRID_SIZE=128
MASKED_NUM=200
UNMASKED_NUM=100
SEED_START=0
HEXT='random'

python -m utils.gen_data_new.py \
    --w $GRID_SIZE \
    --field-mode $HEXT \
    --seed-start $SEED_START \
    --masked $MASKED_NUM \
    --unmasked $UNMASKED_NUM \
    --output-root ./Dataset/rate_change
