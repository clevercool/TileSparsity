#!/bin/bash
DATASET_DIR= "/your/data/path/"
GPU_ID=${1}
pruning_type="ew"
batch_size=128
finetune_steps=5000
mini_finetune_steps=5000
score_type="weight"
#score_type="taylor"
init_checkpoint=""
pre_masks_dir=""
comment=${2}
hostname=`hostname`

log_output_dir="./${hostname}_${pruning_type}_${batch_size}_${finetune_steps}_${mini_finetune_steps=10000}_${score_type}_gpu_${GPU_ID}_${comment}.log"
    command="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python pruning.py \
        --batch_size=$batch_size \
        --finetune_steps=${finetune_steps} \
        --mini_finetune_steps=${mini_finetune_steps} \
        --score_type=${score_type} \
        --pruning_type=${pruning_type} \
        --init_checkpoint=${init_checkpoint} \
        --pre_masks_dir=${pre_masks_dir} \
        --data_dir=${DATASET_DIR} \
        2>&1 | tee ${log_output_dir}
    "
echo $command
echo $log_output_dir
eval $command
