#!/bin/bash

LOSS_TYPE="gradient_tensor_rate"
ALPHA=0.5
TORQUE_LAMBDA=0.1
EPOCHS=1000


python ./train.py --batch-size 100 \
                --lr 0.005 \
                --epochs $EPOCHS \
                --kc 16 \
                --inch 6 \
                --ntrain 300 \
                --ntest 20 \
                --gpu 0 \
                --ex 3.0 \
                --alpha $ALPHA \
                --loss_type $LOSS_TYPE \
                --torque-lambda $TORQUE_LAMBDA