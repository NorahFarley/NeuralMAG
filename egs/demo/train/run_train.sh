#!/bin/bash

LOSS_TYPE="demag_torque_rate"
ALPHA=0
TORQUE_LAMBDA=0.1
EPOCHS=1



python ./train.py --batch-size 100 \
                --lr 0.005 \
                --epochs $EPOCHS \
                --kc 16 \
                --inch 6 \
                --ntrain 300 \
                --ntest 20 \
                --gpu 0 \
                --ex 1.0 \
                --alpha $ALPHA \
                --loss_type $LOSS_TYPE \
                --torque-lambda $TORQUE_LAMBDA
