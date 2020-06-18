#!/usr/bin/env bash

for type in "ew" "bw" "vw" "tw1"; do

task_name='nmt_iwslt15_1'
cuda_device=0
granularity=32
pruning_type=4
STARTTIME=$( date '+%F')
#STARTTIME="2020-04-14"
previous_spar=0
num_train_epochs=1
pruning_type=${type}

for spar in 25 30 37 50 56 62 68 75 81 87 93 ; do

    output_dir="./${task_name}_output/pruning_${spar}%_granularity_${granularity}_type_${pruning_type}_${STARTTIME}"
    init_dir="./envi_model_1/"
    mkdir -p ${output_dir}
    if [[ "${previous_spar}" != "0" ]]
    then
        cp ${init_dir}/model.* ${init_dir}/checkpoint ${output_dir}/
    fi

    command="
        CUDA_VISIBLE_DEVICES=${cuda_device} python -m  nmt.nmt \
        --src=en --tgt=vi \
        --ckpt=${init_dir} \
        --hparams_path=nmt/standard_hparams/iwslt15.json \
        --vocab_prefix=./data_set/vocab \
        --train_prefix=./data_set/train \
        --dev_prefix=./data_set/tst2012 \
        --test_prefix=./data_set/tst2013 \
        --out_dir=${output_dir}/ \
        --override_loaded_hparams=true \
        --sparsity=${spar} \：q
        --pruning_type=${pruning_type} \
        2>&1 | tee ${output_dir}/log.txt 
    "
    echo $command | tee ${output_dir}/cmd.txt
    eval $command

    # command="./run_eval.sh ${output_dir}/translate.ckpt-200 2>&1 | tee ${output_dir}/eval_log.txt"
    # eval $command
    previous_spar=${spar}
done
done
