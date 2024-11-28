#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
root_dir=Experiments/snr0-10/S4Compact/MSE/no_dropout_no_weight_decay;


for test_config in $(find config/dataset_config/test/ -name "*.json"); do
    python test.py --config ${root_dir}/config.json \
        --testset-config $test_config \
        --checkpoint-path ${root_dir}/checkpoints/best.ckpt 
done