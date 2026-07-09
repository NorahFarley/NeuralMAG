#!/bin/bash

loss_type='gradient'
model='gradient_model_324.4.pt'


python -m continue_train --batch-size 100 \
                --lr 0.005 \
                --epochs 900 \
                --kc 16 \
                --inch 6 \
                --ntrain 301 \
                --ntest 20 \
                --gpu 0 \
                --ex 1.0 \
                --loss_type $loss_type \
                --model $model