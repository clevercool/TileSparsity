Pruning VGG
===================

Based on [Tensorflow Model Garden](https://github.com/tensorflow/models).

## Preparing the ImageNet
You need to download the ImageNet dataset and convert it to TFRecord format. See the [script](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) for details. Set the `DATASET_DIR` in `run.sh`.

## Create a virtual environment

```bash
conda env create -f environment.yml
conda activate pruning
```

## Run

```
python download_model.py
./run.sh
```
## References

[Tensorflow Model Garden](https://github.com/tensorflow/models)