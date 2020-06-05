NMT Pruning
===
Based on the google [NMT](https://github.com/tensorflow/nmt).

## Pruning

Create a virtual environment
```bash
conda env create -f environment.yml
conda activate pruning
```
Download the dataset
```
nmt/scripts/download_iwslt15.sh ./data_set
```
Download the model
```
wget http://download.tensorflow.org/models/nmt/envi_model_1.zip
unzip envi_model_1.zip
echo "model_checkpoint_path: \"translate.ckpt\"" > envi_model_1/checkpoint
```

Run the evaluation
```
./run_eval.sh ./envi_model_1/translate.ckpt
```
You will get BLEU: 26.1.

Run the pruning
```
./run.sh
```


For tutorial, you can see [github](https://github.com/tensorflow/nmt/).
# References

[NMT](https://github.com/tensorflow/nmt)
