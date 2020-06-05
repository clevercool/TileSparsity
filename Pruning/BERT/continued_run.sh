# Use for multiple stage pruning. 
#!/usr/bin/env bash
# mode: pretrain or prune
# cuda_device: use which GPU, 0 or 1 or ...
# task_name: MRPC or MNLI or ...
bert_base_dir='../Model/uncased_L-12_H-768_A-12'
task_name='MNLI'
mode=${1}
cuda_device=${2}
granularity=24
pruning_type=24
pretrained_path=../Model/${task_name}_pretrained_model
STARTTIME=$( date '+%F')
#STARTTIME="2020-04-14"
previous_spar=0
number=10
dim=64
g1percent=5
num_train_epochs=1
if [[ "${cuda_device}" = "0" || "${cuda_device}" = "1" || "${cuda_device}" = "2" || "${cuda_device}" = "3" ]]
then
	for spar in 25 35 50 55 60 65 70 75 80 85 90 95; do
	for run in 0  ; do
    if [ "$mode" = "pretrain" ]
    then
        is_pruning_mode=false
		output_dir=${pretrained_path}/model_${run}
        init_dir=$bert_base_dir/bert_model.ckpt
        mkdir -p ${output_dir}
    elif [ "$mode" = "prune" ]
    then
        is_pruning_mode=true
		output_dir="${task_name}_gpu${cuda_device}_pruning_${spar}%_granularity_${granularity}_taylor_score_pruning_type_${pruning_type}_${STARTTIME}_${number}_g1p_${g1percent}_epoch_${num_train_epochs}/model_${run}"
		if [[ "${previous_spar}" = "0" ]]
		then
			init_dir=${pretrained_path}/model_${run}		
		else
			init_dir="${task_name}_gpu${cuda_device}_pruning_${previous_spar}%_granularity_${granularity}_taylor_score_pruning_type_${pruning_type}_${STARTTIME}_${number}_g1p_${g1percent}_epoch_${num_train_epochs}/model_${run}"
		fi

		mkdir -p ${output_dir}
		cp ${init_dir}/*.tf_record ${output_dir}
		if [[ "${previous_spar}" != "0" ]]
		then
		cp ${init_dir}/model.* ${init_dir}/checkpoint ${output_dir}/
		fi
    fi

    command="
	CUDA_VISIBLE_DEVICES=${cuda_device} python3 pruning_classifier.py --task_name=${task_name} \
	--do_train=true \
	--do_eval=true \
	--data_dir=../glue_data/${task_name} \
	--vocab_file=$bert_base_dir/vocab.txt \
	--bert_config_file=$bert_base_dir/bert_config.json \
	--init_checkpoint=${init_dir} \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=${num_train_epochs} \
	--sparsity=$spar \
	--mode=${mode} \
    --is_multistage=False \
	--output_dir=${output_dir} \
	--granularity=${granularity} \
    --pruning_type=${pruning_type} \
    --block_remain=${number} \
    --skip_block_dim=${dim} \
    --g1percent=${g1percent} \
	2>&1 | tee ${output_dir}/log.txt
    "
    echo $command | tee ${output_dir}/cmd.txt
    eval $command
	done
	previous_spar=${spar}
	done
else
    echo "No use any GPU. Need to use one GPU."
fi
