#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
root_dir=Experiments/snr-6-8/CRN/MSE/dropout_20_no_weight_decay;
#root_dir=Experiments/snr7/S4Compact/MSE/dropout_15_no_weight_decay;


for test_config in $(find config/dataset_config/test/ -name "*.json"); do
    echo ${test_config}
    python test.py --config ${root_dir}/config.json \
        --testset-config $test_config \
        --checkpoint-path ${root_dir}/checkpoints/best.ckpt 
done
