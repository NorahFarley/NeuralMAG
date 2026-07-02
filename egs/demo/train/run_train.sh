#!/bin/bash




python ./train.py --batch-size 16 \
                --lr 0.002 \
                --epochs 1000 \
                --kc 16 \
                --inch 6 \
                --ntrain 301 \
                --ntest 20 \
                --gpu 0 \
                --ex 1.0 \
                --alpha 0.5 \
                --loss_type "baseline"
