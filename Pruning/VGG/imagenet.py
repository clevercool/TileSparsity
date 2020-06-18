# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
# from slim.preprocessing import inception_preprocessing

import imagenet_preprocessing
from imagenet_preprocessing import preprocess_image
__all__ = ["train", "test", "normalize", "normalize_alexnet"]

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {"train": 1281167, "validation": 50000}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500

_CHANNEL_MEANS = imagenet_preprocessing._CHANNEL_MEANS


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, "train-%05d-of-01024" % i)
            for i in range(_NUM_TRAIN_FILES)
        ]
    else:
        return [
            os.path.join(data_dir, "validation-%05d-of-00128" % i) for i in range(128)
        ]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/class/label": tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        "image/class/text": tf.FixedLenFeature([], dtype=tf.string, default_value=""),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {
            k: sparse_float32
            for k in [
                "image/object/bbox/xmin",
                "image/object/bbox/ymin",
                "image/object/bbox/xmax",
                "image/object/bbox/ymax",
            ]
        }
    )

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features["image/class/label"], dtype=tf.int32)

    xmin = tf.expand_dims(features["image/object/bbox/xmin"].values, 0)
    ymin = tf.expand_dims(features["image/object/bbox/ymin"].values, 0)
    xmax = tf.expand_dims(features["image/object/bbox/xmax"].values, 0)
    ymax = tf.expand_dims(features["image/object/bbox/ymax"].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features["image/encoded"], label, bbox


def parse_record(raw_record):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.

    Returns:
      Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    # label = slim.one_hot_encoding(
    #        label, 1001)
    # label = tf.one_hot(tf.reshape(label, shape=[]), _NUM_CLASSES)
    label = tf.reshape(label, shape=[])

    return (image_buffer, bbox), label


def normalize(x: np.ndarray) -> np.ndarray:
    return x - np.array(_CHANNEL_MEANS, dtype=np.float32)


def normalize_alexnet(x: np.ndarray) -> np.ndarray:
    return (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )


def recover_from_alexnet(x: np.ndarray) -> np.ndarray:
    return (x * np.array([0.229, 0.224, 0.225], dtype=np.float32)) + np.array(
        [0.485, 0.456, 0.406], dtype=np.float32
    )


# def inception_preprocess_image(
#     image_buffer,
#     output_height,
#     output_width,
#     num_channels,
#     is_training=False,
#     bbox=None,
#     normed=True,
#     fast_mode=True,
#     add_image_summaries=True,
# ):
#     return inception_preprocessing.preprocess_image(
#         image=tf.image.decode_jpeg(image_buffer, channels=num_channels),
#         height=output_height,
#         width=output_width,
#         bbox=bbox,
#         is_training=is_training,
#     )


def input_fn(
    is_training,
    data_dir,
    batch_size,
    image_size=_DEFAULT_IMAGE_SIZE,
    num_epochs=1,
    num_parallel_calls=1,
    multi_gpu=False,
    is_shuffle=False,
    pre_transform_fn=None,
    transform_fn=None,
    preprocessing_fn=None,
    normed=True,
):
    """Input function which provides batches for train or eval.
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.
      multi_gpu: Whether this is run multi-GPU. Note that this is only required
        currently to handle the batch leftovers, and can be removed
        when that is handled directly by Estimator.

    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training and is_shuffle:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    num_images = is_training and _NUM_IMAGES["train"] or _NUM_IMAGES["validation"]

    # Convert to individual records
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    return process_record_dataset(
        dataset,
        is_training and is_shuffle,
        image_size,
        batch_size,
        _SHUFFLE_BUFFER,
        parse_record,
        num_epochs,
        num_parallel_calls,
        examples_per_epoch=num_images,
        multi_gpu=multi_gpu,
        pre_transform_fn=pre_transform_fn,
        transform_fn=transform_fn,
        preprocessing_fn=preprocessing_fn,
    )



def parse_record_t(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = _parse_example_proto(raw_record)

  image = preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=224,
      output_width=224,
      num_channels=3,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label

def input_function(
             is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record_t,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False):
    """Input function which provides batches for train or eval.

    Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: The directory containing the input data.
        batch_size: The number of samples per batch.
        num_epochs: The number of epochs to repeat the dataset.
        dtype: Data type to use for images/features
        datasets_num_private_threads: Number of private threads for tf.data.
        parse_record_fn: Function to use for parsing the records.
        input_context: A `tf.distribute.InputContext` object passed in by
        `tf.distribute.Strategy`.
        drop_remainder: A boolean indicates whether to drop the remainder of the
        batches. If True, the batch dimension will be static.
        tf_data_experimental_slack: Whether to enable tf.data's
        `experimental_slack` option.

    Returns:
        A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if input_context:
        logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
            input_context.input_pipeline_id, input_context.num_input_pipelines)
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 10 means that up to 10 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return resnet_util.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record_fn,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=tf_data_experimental_slack,
    )

def input_fn_train(
    data_dir,
    batch_size,
    num_epochs):
    return input_function(
        is_training=True,
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=num_epochs)

def input_fn_eval(
    data_dir,
    batch_size):
    return input_function(
        is_training=False,
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=1)

def process_record_dataset(
    dataset,
    is_training,
    image_size,
    batch_size,
    shuffle_buffer,
    parse_record_fn,
    num_epochs=1,
    num_parallel_calls=1,
    examples_per_epoch=0,
    multi_gpu=False,
    pre_transform_fn=None,
    transform_fn=None,
    preprocessing_fn=None,
    normed=True,
):
    """Given a Dataset with raw records, parse each record into images and labels,
    and return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.
      examples_per_epoch: The number of examples in the current set that
        are processed each epoch. Note that this is only used for multi-GPU mode,
        and only to handle what will eventually be handled inside of Estimator.
      multi_gpu: Whether this is run multi-GPU. Note that this is only required
        currently to handle the batch leftovers (see below), and can be removed
        when that is handled directly by Estimator.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """

    if pre_transform_fn is not None:
        dataset = pre_transform_fn(dataset)

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to ensure
        # that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat the
    # dataset for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)
    #dataset = dataset.repeat()

    # Currently, if we are using multiple GPUs, we can't pass in uneven batches.
    # (For example, if we have 4 GPUs, the number of examples in each batch
    # must be divisible by 4.) We already ensured this for the batch_size, but
    # we have to additionally ensure that any "leftover" examples-- the remainder
    # examples (total examples % batch_size) that get called a batch for the very
    # last batch of an epoch-- do not raise an error when we try to split them
    # over the GPUs. This will likely be handled by Estimator during replication
    # in the future, but for now, we just drop the leftovers here.
    if multi_gpu:
        total_examples = num_epochs * examples_per_epoch
        dataset = dataset.take(batch_size * (total_examples // batch_size))

    # Parse the raw records into images and labels
    dataset = dataset.map(parse_record_fn, num_parallel_calls=num_parallel_calls)

    if transform_fn is not None:
        dataset = transform_fn(dataset)

    dataset = dataset.map(
        lambda image_buffer_with_bbox, label: (
            (preprocessing_fn or imagenet_preprocessing.preprocess_image)(
                image_buffer=image_buffer_with_bbox[0],
                bbox=image_buffer_with_bbox[1],
                output_height=image_size,
                output_width=image_size,
                num_channels=_NUM_CHANNELS,
                is_training=is_training,
            ),
            label,
        )
    )

    dataset = dataset.batch(batch_size)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path.
    dataset = dataset.prefetch(2)
    
    
    #dataset = dataset.take(50)

    return dataset


def train(
    data_dir,
    batch_size=64,
    image_size=_DEFAULT_IMAGE_SIZE,
    num_parallel_calls=1,
    num_epochs=1,
    multi_gpu=False,
    is_shuffle=False,
    pre_transform_fn=None,
    transform_fn=None,
    preprocessing_fn=None,
    normed=True,
):
    return input_fn(
        True,
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_parallel_calls=num_parallel_calls,
        num_epochs=num_epochs,
        multi_gpu=multi_gpu,
        is_shuffle=is_shuffle,
        pre_transform_fn=pre_transform_fn,
        transform_fn=transform_fn,
        preprocessing_fn=preprocessing_fn,
        normed=normed,
    )


def test(
    data_dir,
    batch_size=1,
    image_size=_DEFAULT_IMAGE_SIZE,
    num_parallel_calls=1,
    multi_gpu=False,
    is_shuffle=False,
    pre_transform_fn=None,
    transform_fn=None,
    preprocessing_fn=None,
    normed=True,
):
    return input_fn(
        False,
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        multi_gpu=multi_gpu,
        is_shuffle=is_shuffle,
        num_parallel_calls=num_parallel_calls,
        pre_transform_fn=pre_transform_fn,
        transform_fn=transform_fn,
        preprocessing_fn=preprocessing_fn,
        normed=normed,
    )
