#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
#model_config=config/model_config/ANN/MSE/no_dropout_no_weight_decay.json
#model_config=config/model_config/S4Compact/MSE/dropout_10_no_weight_decay.json
dataset_config="config/dataset_config/train/snr7.json"

for model_config in $(find config/model_config/ANN/MAE/ -name "no_dropout*.json"); do

        python train.py --model-config ${model_config} \
                --dataset-config $dataset_config \
                --gpus 1 
done
