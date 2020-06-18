#!/usr/bin/env bash
# mode: pretrain or prune
# cuda_device: use which GPU, 0 or 1 or ...
# task_name: MRPC or MNLI or ...
bert_base_dir='./Model/uncased_L-12_H-768_A-12'
task_name='MRPC'
mode=${1}
cuda_device=${2}
granularity=128
pruning_type=5
pretrained_path=./Model/${task_name}_pretrained_model
STARTTIME=$( date '+%F')
if [[ "${cuda_device}" = "0" || "${cuda_device}" = "1" || "${cuda_device}" = "2" ]]
then
    for spar in 50 ; do
    for run in 0 ; do
    if [ "$mode" = "pretrain" ]
    then
        output_dir=${pretrained_path}/model_${run}
        init_dir=$bert_base_dir/bert_model.ckpt
        mkdir -p ${output_dir}
    elif [ "$mode" = "prune" ]
    then
        output_dir="output/${task_name}_pruning_${spar}%_granularity_${granularity}_taylor_score_pruning_type_${pruning_type}_${STARTTIME}/model_${run}"
        init_dir=${pretrained_path}/model_${run}
        mkdir -p ${output_dir}
        cp ${init_dir}/*.tf_record ${output_dir}
    elif [ "$mode" = "export" ]
    then
        init_dir=${pretrained_path}/model_${run}
        output_dir="${task_name}_pruning_${spar}%_${STARTTIME}/model_${run}"
    fi
    command="
    CUDA_VISIBLE_DEVICES=${cuda_device} python3 pruning_classifier.py --task_name=${task_name} \
    --do_train=true \
    --do_eval=true \
    --data_dir=./glue_data/  \
    --vocab_file=$bert_base_dir/vocab.txt \
    --bert_config_file=$bert_base_dir/bert_config.json \
    --init_checkpoint=${init_dir} \
    --max_seq_length=128 \
    --train_batch_size=16 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --sparsity=$spar \
    --mode=${mode} \
    --is_multistage=false \
    --output_dir="${output_dir}" \
    --granularity=${granularity} \
    --pruning_type=${pruning_type} \
    2>&1 | tee ${output_dir}/log.txt
    "
    echo $command | tee ${output_dir}/cmd.txt
    eval $command
    done
    done
else
    echo "No use any GPU. Need to use one GPU."
fi
