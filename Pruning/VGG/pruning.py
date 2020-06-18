# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import tensorflow as tf
import numpy as np
import pickle
import imagenet
import os
import sys
import time
import copy

from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16 as slim_vgg_16
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_arg_scope
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2 as slim_alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2_arg_scope
from pathlib import Path
from myhook import PruningHook
from utils import root_dir, ensure_dir
from functools import partial
from imagenet_preprocessing import alexnet_preprocess_image, preprocess_image
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "pre_masks_dir", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "arch_name", "vgg16",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer("batch_size", 32, "Total batch size.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 100,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("finetune_steps", 10000,
                     "How many steps to make in each finetune call.")

flags.DEFINE_integer("mini_finetune_steps", 5000,
                     "How many steps to make in each mini_finetune call.")

flags.DEFINE_integer("pruning_steps", 5,
                     "How many steps to make in each estimator call.")
                     
flags.DEFINE_string("score_type", "weight",
                     "How many steps to make in each pruning call.")

flags.DEFINE_float(
    "sparsity", 0,
    "The sparsity of network pruning. ")

flags.DEFINE_integer(
    "granularity", 128,
    "The granularity for granularity pruning. ")

flags.DEFINE_string(
    "pruning_type", "ew",
    "The pruning_type for network pruning. ")

flags.DEFINE_integer(
    "block_remain", 0,
    "The block_remain for network pruning. ")

flags.DEFINE_integer(
    "skip_block_dim", 0,
    "The skip_block_dim for network pruning. ")

flags.DEFINE_float(
    "g1percent", 0,
    "G = 1 percent ")

input_shapes = {
    "vgg16": (1, 224, 224, 3),
    "vgg19": (1, 224, 224, 3),
    "inception_v1": (1, 224, 224, 3),
    "inception_v3": (1, 299, 299, 3),
    "resnet_v1_50": (1, 224, 224, 3),
    "resnet_v1_152": (1, 224, 224, 3),
    "resnet_v2_50": (1, 299, 299, 3),
    "resnet_v2_101": (1, 299, 299, 3),
    "resnet_v2_152": (1, 299, 299, 3),
    "resnet_v2_200": (1, 299, 299, 3),
    "mobilenet_v1_1.0": (1, 224, 224, 3),
    "mobilenet_v2_1.0_224": (1, 224, 224, 3),
    "inception_resnet_v2": (1, 299, 299, 3),
    "nasnet-a_large": (1, 331, 331, 3),
    "facenet": (1, 160, 160, 3),
    "rnn_lstm_gru_stacked": (1, 150),
}

class VGG16:
    def __call__(self, inputs, training=False):
        with slim.arg_scope(vgg_arg_scope()):
            return slim_vgg_16(inputs, is_training=training)[0]
# class Alexnet:
#     def __call__(self, inputs, training=False):
#         with slim.arg_scope(alexnet_v2_arg_scope()):
#             return slim_alexnet_v2(inputs, is_training=training)[0]

class AlexNet:
    """Class that defines a graph to recognize digits in the MNIST dataset."""

    def __init__(self, data_format: str = "channels_first"):
        """Creates a model for classifying a hand-written digit.

        Args:
          data_format: Either "channels_first" or "channels_last".
            "channels_first" is typically faster on GPUs while "channels_last" is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        self.conv1 = tf.layers.Conv2D(
            filters=64,
            kernel_size=11,
            strides=4,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv2 = tf.layers.Conv2D(
            filters=192,
            kernel_size=5,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv3 = tf.layers.Conv2D(
            filters=384,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv4 = tf.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.conv5 = tf.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format=data_format,
            activation=tf.nn.relu,
        )
        self.fc1 = tf.layers.Dense(4096, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(4096, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(1000)
        self.max_pool2d = tf.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), data_format=data_format
        )
        self.dropout = tf.layers.Dropout(rate=0.5)

    def __call__(self, inputs, training=False):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        y = self.conv1(inputs)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.max_pool2d(y)
        y = tf.layers.Flatten()(y)
        y = self.dropout(y, training=training)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        y = self.fc2(y)
        return self.fc3(y)


def model_fn(features, labels, mode, pruning_hook, is_pruning):
    """The model_fn argument for creating an Estimator."""
    model = VGG16()
    if FLAGS.arch_name == "alexnet":
        model = AlexNet()
    image = features
    
    # Generate a summary node for the images
    tf.summary.image("images", features, max_outputs=6)

    logits = model(image, mode == tf.estimator.ModeKeys.TRAIN)

    if pruning_hook is not None:
        #pruning_hook.print_weights()
        pruning_hook.is_pruning = is_pruning
        pruning_hook.insert_masks()

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def loss_filter_fn(name):
        return "batch_normalization" not in name

    weight_decay = 1e-4
    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if loss_filter_fn(v.name)]
    )

    #[print(t.name.find("/weights:0")) for op in tf.get_default_graph().get_operations() for t in op.values()]
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = 1e-4
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))
    metrics = {'accuracy': accuracy,
        'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
    tf.compat.v1.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
    )


def main(_):
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

    arch_name = FLAGS.arch_name
    
    if FLAGS.init_checkpoint == None or FLAGS.init_checkpoint == "":
        pretrained_model_dir = root_dir() / "downloads" / "model" / arch_name
    else:
        pretrained_model_dir = FLAGS.init_checkpoint


    if FLAGS.pre_masks_dir == None  or FLAGS.init_checkpoint == "":
        all_mask_values = {}
    else:
        with open(FLAGS.pre_masks_dir, "rb") as file:
            all_mask_values = pickle.load(file)

    if FLAGS.output_dir == None:
        model_dir = root_dir() / "train" / arch_name / now
    else:
        model_dir = root_dir() / "train" / FLAGS.output_dir
    
    if FLAGS.data_dir != None:
        data_dir = FLAGS.data_dir
    else:
        raise ValueError(
        "data_dir must set")

    if FLAGS.pruning_type != "":
        pruning_type = FLAGS.pruning_type
    else:
        raise ValueError(
        "pruning_type must set")

    tf.logging.info("Pruning_type : %s\n"% pruning_type)

    if "tw" in pruning_type or "bw" in pruning_type:
        masks=[
            "vgg_16/conv1/conv1_1/weights:0",
            "vgg_16/conv1/conv1_2/weights:0",
            "vgg_16/conv2/conv2_1/weights:0",
            "vgg_16/conv2/conv2_2/weights:0",
            "vgg_16/conv3/conv3_1/weights:0",
            "vgg_16/conv3/conv3_2/weights:0",
            "vgg_16/conv3/conv3_3/weights:0",
            "vgg_16/conv4/conv4_1/weights:0",
            "vgg_16/conv4/conv4_2/weights:0",
            "vgg_16/conv4/conv4_3/weights:0",
            "vgg_16/conv5/conv5_1/weights:0",
            "vgg_16/conv5/conv5_2/weights:0",
            "vgg_16/conv5/conv5_3/weights:0",
            # "vgg_16/fc6/weights:0",
            # "vgg_16/fc7/weights:0",
            # "vgg_16/fc8/weights:0",
            ]
    else:
        masks=[
            "vgg_16/conv1/conv1_1/weights:0",
            "vgg_16/conv1/conv1_2/weights:0",
            "vgg_16/conv2/conv2_1/weights:0",
            "vgg_16/conv2/conv2_2/weights:0",
            "vgg_16/conv3/conv3_1/weights:0",
            "vgg_16/conv3/conv3_2/weights:0",
            "vgg_16/conv3/conv3_3/weights:0",
            "vgg_16/conv4/conv4_1/weights:0",
            "vgg_16/conv4/conv4_2/weights:0",
            "vgg_16/conv4/conv4_3/weights:0",
            "vgg_16/conv5/conv5_1/weights:0",
            "vgg_16/conv5/conv5_2/weights:0",
            "vgg_16/conv5/conv5_3/weights:0",
            # "vgg_16/fc6/weights:0",
            # "vgg_16/fc7/weights:0",
            # "vgg_16/fc8/weights:0",
            ]
    if arch_name == "alexnet":
        masks=[
            "conv2d/kernel:0",
            "conv2d_1/kernel:0",
            "conv2d_2/kernel:0",
            "conv2d_3/kernel:0",
            "conv2d_4/kernel:0",
            # "dense/kernel:0",
            # "dense_1/kernel:0",
            # "dense_2/kernel:0",
            ]
            
    batch_size = int(FLAGS.batch_size)
    num_epochs = int(FLAGS.num_train_epochs)
    if FLAGS.score_type == "weight":
        is_masked_weight = True
    else:
        is_masked_weight = False
    pruning_steps = int(FLAGS.pruning_steps)
    finetune_steps = int(FLAGS.finetune_steps)
    mini_finetune_steps = int(FLAGS.mini_finetune_steps)
    if is_masked_weight:
        tf.logging.info("Pruning score : Weight ! \n")
    else:
        tf.logging.info("Pruning score : Taylor ! \n")

    if "tw" in pruning_type:
        pruning_layers = [[0,1],[2,3],[4,5,6],[7,8,9],[10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]
    if "ew" in pruning_type:
        pruning_layers = [[0,1,2,3,4,5,6,7,8,9,10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]
    if "vw" in pruning_type:
        pruning_layers = [[0,1],[2,3],[4,5,6],[7,8,9],[10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]

    if "bw" in pruning_type:
        pruning_layers = [[0,1],[2,3],[4,5,6],[7,8,9],[10,11,12]]
        sparsity_stages = [25, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96]


    early_stop = [100, 100, 100, 100, 100, 100]

    previous_checkpoint = tf.train.latest_checkpoint(pretrained_model_dir)
    previous_mask_values = copy.deepcopy(all_mask_values)

    # Set hooks
    pruning_hook = PruningHook(masks, pruning_type, is_masked_weight, previous_mask_values)

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {
        "train_accuracy": "train_accuracy",
        "train_accuracy_top_5": "train_accuracy_top_5",
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100
    )

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    estimator_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_secs=60*60*24*5, # 5 days
        save_checkpoints_steps=None,
        keep_checkpoint_max = len(sparsity_stages) + 3,
        )


    if arch_name == "vgg16":
        pre_fn = preprocess_image
    if arch_name == "alexnet":
        pre_fn = alexnet_preprocess_image

    # Set input_fn
    def eval_input_fn():
        return imagenet.test(
            data_dir,
            batch_size,
            num_parallel_calls=40,
            transform_fn=lambda ds: ds.map(lambda image, label: (image, label - 1)),
            preprocessing_fn = pre_fn
        )

    def train_input_fn():
        return imagenet.train(
            data_dir,
            batch_size,
            is_shuffle=True,
            num_epochs=num_epochs,
            num_parallel_calls=40,
            transform_fn=lambda ds: ds.map(lambda image, label: (image, label - 1)),
            preprocessing_fn = pre_fn
        )


    warm_start_settings = tf.estimator.WarmStartSettings(previous_checkpoint)
    estimator = tf.estimator.Estimator(
        model_fn=partial(
            model_fn,
            pruning_hook=None,
            is_pruning=False
        ),
        config=estimator_config,
        warm_start_from=warm_start_settings,
    )

    tf.logging.info('First Evaluation !!!')
    eval_results = estimator.evaluate(
        input_fn=eval_input_fn
    )
    tf.logging.info('Done First Evaluation !!!\n')
    tf.logging.info("First Evaluation results:\n\n\t%s\n" % eval_results)

    ## Multi stages to prune the CNN model
    for stage in range(len(sparsity_stages)):
        stage_dir = model_dir / ("sparsity_stage_" + str(sparsity_stages[stage]))
        
        sparsity = sparsity_stages[stage]
        pruning_hook.sparsity = sparsity

        tf.logging.info("\n\n\n\n\n\n\n\n\n\n\n\n")
        tf.logging.info("Stage : %d, Sparsity : %f, Finetune_steps : %d" %(stage, sparsity, finetune_steps))
        tf.logging.info("early_stop:")
        tf.logging.info(early_stop)

        is_early_stop = True
        for layer in range(len(pruning_layers)):
            # if "ew" in pruning_type:
            #     sparsity = sparsity_layers[layer]
            pruning_hook.sparsity = sparsity
            tf.logging.info("\n\n\n\n\n")
            if early_stop[layer] < sparsity:
                continue
            is_early_stop = False

            layer_dir = stage_dir / ("layer_" + str(layer))
            masks_dir = layer_dir / "masks"
            os.makedirs(masks_dir)

            masks_now = pruning_layers[layer]

            assert(previous_checkpoint)
            warm_start_settings = tf.estimator.WarmStartSettings(previous_checkpoint)
            pruning_hook.masks_now = masks_now
            pruning_hook.mask_values = copy.deepcopy(previous_mask_values)

            # Set estimator
            pruning_estimator = tf.estimator.Estimator(
                model_fn=partial(
                    model_fn,
                    pruning_hook=pruning_hook,
                    is_pruning=True
                ),
                model_dir=layer_dir,
                warm_start_from=warm_start_settings,
                config=estimator_config,
            )
            
            tf.logging.info('Prune !!!')
            pruning_hook.initialize(pruning_steps)
            pruning_estimator.train(
                input_fn=train_input_fn, 
                hooks=[logging_hook, pruning_hook],
                steps=pruning_steps
            )
            tf.logging.info('Done Prune !!!\n')

            # Set fine-tune estimator
            # The mask_values must be updated by estimator_pruning.train().
            # The mask_values in estimator is constant, 
            # then we build the estimator in every iteration for fine-tune.
            estimator = tf.estimator.Estimator(
                model_fn=partial(
                    model_fn,
                    pruning_hook=pruning_hook,
                    is_pruning=False
                ),
                model_dir=layer_dir,
                config=estimator_config,
            )

            mini_train_times = 0
            while(True):
                tf.logging.info('Mini Fine Tune !!!')
                estimator.train(
                    input_fn=train_input_fn, 
                    hooks=[logging_hook],
                    steps=mini_finetune_steps
                )
                tf.logging.info('Done Mini Fine Tune !!!\n')


                tf.logging.info('Evaluation !!!')
                eval_results = estimator.evaluate(
                    input_fn=eval_input_fn,
                )

                tf.logging.info("Pruning Sparsity %d Layer %d Evaluation results:\n\n\t%s\n" % (sparsity, layer, eval_results))
                tf.logging.info('Done Evaluation !!!\n')

                
                threshold = 0.8885
                if sparsity > 75:
                    threshold = 0.88818
                if sparsity > 85:
                    threshold = 0.0

                if arch_name == "alexnet":
                    threshold = 0.775

                if eval_results["accuracy_top_5"] > threshold:
                    previous_checkpoint = tf.train.latest_checkpoint(layer_dir)
                    previous_mask_values = copy.deepcopy(pruning_hook.mask_values)
                    tf.logging.info('GOOD CHECKPOINT!!!\n')
                    with open(masks_dir / ("goog_mask_" + str(sparsity) + ".pkl"), "wb") as file:
                        pickle.dump(pruning_hook.mask_values, file)

                    break
                else:
                    tf.logging.info('BAD CHECKPOINT!!!\n')
                    with open(masks_dir / ("bad_mask_" + str(sparsity) + ".pkl"), "wb") as file:
                        pickle.dump(pruning_hook.mask_values, file)

                mini_train_times += 1

                if(mini_train_times > 8):
                    early_stop[layer] = sparsity
                    break

        if is_early_stop:
            break
        
        tf.logging.info("\n\n\n\n\n")
        all_layer_dir = stage_dir / "all_layers"
        masks_dir = all_layer_dir / "masks"
        os.makedirs(masks_dir)

        assert(previous_checkpoint)
        warm_start_settings = tf.estimator.WarmStartSettings(previous_checkpoint)
        pruning_hook.mask_values = copy.deepcopy(previous_mask_values)

        # Set fine-tune estimator
        # The mask_values must be updated by estimator_pruning.train().
        # The mask_values in estimator is constant, 
        # then we build the estimator in every iteration for fine-tune.
        estimator = tf.estimator.Estimator(
            model_fn=partial(
                model_fn,
                pruning_hook=pruning_hook,
                is_pruning=False,
            ),
            model_dir=all_layer_dir,
            config=estimator_config,
            warm_start_from=warm_start_settings,
        )

        tf.logging.info('Fine Tune !!!')
        estimator.train(
            input_fn=train_input_fn, 
            hooks=[logging_hook],
            steps=finetune_steps
        )
        tf.logging.info('Done Fine Tune !!!\n')

        tf.logging.info('Last Evaluation !!!')
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn
        )
        tf.logging.info('Done Evaluation !!!\n')
        tf.logging.info("After Pruning Evaluation results:\n\n\t%s\n" % eval_results)

        previous_checkpoint = tf.train.latest_checkpoint(all_layer_dir)
        previous_mask_values = copy.deepcopy(pruning_hook.mask_values)
        with open(masks_dir / ("mask_" + str(sparsity) + ".pkl"), "wb") as file:
            pickle.dump(pruning_hook.mask_values, file)

    return

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
