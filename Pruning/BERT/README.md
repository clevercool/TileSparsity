BERT pruning
===
## Introduction
This project is based on the [BERT](https://github.com/google-research/bert), where you can find more information.

Currently, we can run the bash file "continued_run.sh" to execute the bert-pruning. The bash file requires two args: mode and GPU device number.
For example, the command of using GPU 0 and prune mode is "bash continued_run.sh prune 0".
Note that the bash file use the bert pretrained model in the parent folder (../Model/), like:

```
/BERT: the folder of this project
   |--/other_function:
   |--/utils:
      |--Myhook.py: The main file for pruning.
   |--pruning_classifier.py
   |--run_squad.py
   |--run.sh           # run the pruning.
   |--run_squad.sh     # run the pruning on SQuAd.
   |--continued_run.sh # run the multiple stage pruning.
   /Model
      |--/uncased_L-12_H-768_A_12: small bert model pretrained by google. 
      |--/MNLI_pretrained_model: use uncased_L-12_H-768_A_12 as pretrained model and fine-tune on MNLI with some epochs.
   /glue_data: 
      |--/MNLI: 
      |--/MRPC:
      |--/...
```

## Pruning
Create a virtual environment
```bash
conda env create -f environment.yml
conda activate pruning
```
Download the dataset
```
# GLUE
python ./other_function/download_glue_data.py
# SQuAD
bash ./download_squad_data.sh
```
Download the model
```
mkdir -p Model
wget -P ./Model https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip ./Model/uncased_L-12_H-768_A-12.zip  -d ./Model/
```

Pretrain the model
```
# For GLUE 
bash ./run.sh pretrain 0
# For SQuAD
bash ./run_squad.sh pretrain 0
```

Prune the model. Set pruning_type to choose the pruning method.

type 0: prune the whole head. For example, prune from 12 heads to 6 heads. 

type 1: prune the size of per head. For example, prune from 64 per head to 32 per head. 

type 2: granularity pruning on dim K. The granularity must be divide hidden units (768 here), like 1, 128, 256, 768. 

type 3: granularity pruning on [Q K V] together. The granularity is 3x768. 

type 4: VW. Prune on N dimension. Block size is 1x16, and choose top 4. 

type 5: BW. Prune on K-N dimension. Block size is 32x32. 

type 7: TW pruning on dim K, consider the whole network but not layer by layer.

type 8: TW pruning on dim K and N, consider the whole network but not layer by layer.
```
# For GLUE 
bash ./continued_run.sh prune 0
# For SQuAD
bash ./run_squad.sh prune 0
```

You can find the output file started with task name.


## References

[BERT](https://github.com/google-research/bert)