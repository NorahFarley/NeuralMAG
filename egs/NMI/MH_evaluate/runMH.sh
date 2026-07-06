#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

gpu=0
model='model.pt'
loss_type='baseline'

for width in 64 96
do
    for mask in True triangle hole
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax 0.5e-6 --Ku 0.0 --dtime 2.0e-13 --mask $mask --loss_type $loss_type --model_name $model
    done
    
    for Ms in 1200 1000 800 600 400
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms $Ms --Ax 0.5e-6 --Ku 0.0 --dtime 2.0e-13 --loss_type $loss_type --model_name $model
    done

    for Ku in 1e5 2e5 3e5 4e5
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax 0.5e-6 --Ku $Ku --Kvec 1,0,0 --dtime 2.0e-13 --loss_type $loss_type --model_name $model
    done
    
    for Ax in 0.7e-6 0.6e-6 0.4e-6 0.3e-6
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax $Ax --Ku 0.0 --dtime 2.0e-13 --loss_type $loss_type --model_name $model
    done
done


for width in 128 256
do

    for Ms in 1200 1000 800 600 400
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms $Ms --Ax 0.5e-6 --Ku 0.0 --dtime 5.0e-13 --max_iter 200000 --loss_type $loss_type --model_name $model
    done

    for Ku in 1e5 2e5 3e5 4e5
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax 0.5e-6 --Ku $Ku --Kvec 1,0,0 --dtime 5.0e-13 --max_iter 200000 --loss_type $loss_type --model_name $model
    done

    for Ax in 0.7e-6 0.6e-6 0.4e-6 0.3e-6
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax $Ax --Ku 0.0 --dtime 5.0e-13 --max_iter 200000 --loss_type $loss_type --model_name $model
    done

    for mask in True triangle hole
    do
        python ./MH_unet_mm.py --gpu $gpu --w $width --layer 2 --Ms 1000 --Ax 0.5e-6 --Ku 0.0 --dtime 5.0e-13 --max_iter 200000 --mask $mask --loss_type $loss_type --model_name $model
    done

done
