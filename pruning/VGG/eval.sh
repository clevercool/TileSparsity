DATASET_DIR/home/yxqiu/data/imagenet
TRAIN_DIR=./train_tmp
CHECKPOINT_PATH=./downloads/model/vgg16
CHECKPOINT_FILE=./downloads/model/vgg16/imagenet_vgg16.ckpt


python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=vgg16