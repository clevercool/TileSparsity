# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import sys
import os
import numpy as np
import tensorflow as tf
from utils import modeling, optimization, tokenization
from utils.data_processor import *
from utils.MyHook import logPerformanceHook, NetworkPruningHook, InitializeGlobalStepHook, exportPrunedModel

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float(
    "sparsity", 0,
    "The sparsity of network pruning. ")

flags.DEFINE_string("mode", None, "[pretrain/prune/export] mode.")

flags.DEFINE_bool("is_multistage", False, "Use pruning mode or not.")

flags.DEFINE_integer(
    "granularity", 128,
    "The granularity for granularity pruning. ")

flags.DEFINE_integer(
    "pruning_type", 0,
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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, isPruning):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    mask_vars = tf.global_variables()
    for var in mask_vars:
      if "mask-o" in var.name:
        tvars.append(var)
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      if isPruning == True:
        train_op = optimization.create_optimizer(
          0, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[logPerformanceHook(logits, label_ids, is_real_example, per_example_loss, num_train_steps)]
          )

    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

def define_train_eval_input_fn():

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mismnli": MisMnliProcessor,
      "mrpc": MrpcProcessor,
      "rte": RteProcessor,
      "sst-2": Sst2Processor,
      "sts-b": StsbProcessor,
      "qqp": QqpProcessor,
      "qnli": QnliProcessor,
      "wnli": WnliProcessor,
  }
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


  train_examples = None
  num_train_steps = None
  num_warmup_steps = None


  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  if(tf.gfile.Exists(train_file) == False):
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  
  eval_examples = processor.get_dev_examples(FLAGS.data_dir)
  num_actual_eval_examples = len(eval_examples)
    
  eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
  if(tf.gfile.Exists(eval_file) == False):
    file_based_convert_examples_to_features(
       eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

  tf.logging.info("***** Running evaluation *****")
  tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                  len(eval_examples), num_actual_eval_examples,
                  len(eval_examples) - num_actual_eval_examples)
  tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

  # This tells the estimator to run through the entire set.
  eval_steps = None
  # However, if running eval on the TPU, you will need to specify the
  # number of steps.
  if FLAGS.use_tpu:
    assert len(eval_examples) % FLAGS.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

  eval_drop_remainder = True if FLAGS.use_tpu else False
  eval_input_fn = file_based_input_fn_builder(
      input_file=eval_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=eval_drop_remainder)

  return label_list, train_input_fn, num_train_steps, eval_input_fn, eval_steps, num_warmup_steps

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  label_list, train_input_fn, num_train_steps, eval_input_fn, eval_steps, num_warmup_steps = define_train_eval_input_fn()

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=2,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      session_config=gpu_config)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      isPruning=False)

  pruning_model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      isPruning=True)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  pruning_estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=pruning_model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)


  def print_evaluating_result(estimator, input_fn, steps, isFirst=False):
    result = estimator.evaluate(input_fn=input_fn, steps=steps)
    file_open_mode = "w" if isFirst == True else "a"

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, file_open_mode) as writer:
      tf.logging.info("***** Eval results *****")
      
      mode = "pruning mode" if (FLAGS.mode == "prune") else "pretrain mode"
      writer.write("***** Eval results under %s *****\n" % (str(mode)))
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
      writer.write("*******************************************\n\n\n")

    output_eval_file = os.path.join(FLAGS.output_dir, "Important_result.txt")
    with tf.gfile.GFile(output_eval_file, file_open_mode) as writer:
      mode = "pruning mode" if (FLAGS.mode == "prune") else "pretrain mode"
      writer.write("***** Eval results under %s *****\n" % (str(mode)))
      for key in sorted(result.keys()):
        writer.write("%s = %s\n" % (key, str(result[key])))
      writer.write("*******************************************\n\n\n")

  if FLAGS.mode == "export":
    estimator.evaluate(input_fn=eval_input_fn, steps=1, hooks=[exportPrunedModel()])
  else:
    # evaluation before pruning
    print_evaluating_result(estimator, eval_input_fn, eval_steps, True)
    
    if FLAGS.mode == "prune":
      pruning_hook = NetworkPruningHook(FLAGS)
      if FLAGS.is_multistage == False:
        pruning_hook.initialize(FLAGS.sparsity, 100, 1)
        pruning_estimator.train(input_fn=train_input_fn, steps=100, hooks=[pruning_hook, InitializeGlobalStepHook()])
        print_evaluating_result(estimator, eval_input_fn, eval_steps)
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)
      else:
        have_trained_step = 0
        if FLAGS.sparsity <= 20: 
          pruning_hook.initialize(FLAGS.sparsity, 50, 1)
          pruning_estimator.train(input_fn=train_input_fn, steps=50, hooks=[pruning_hook])
          print_evaluating_result(estimator, eval_input_fn, eval_steps)
        else: 
          sparsity_schedule = range(20, FLAGS.sparsity+1, 2)
          pruning_step = 100  if num_train_steps > 3100*len(sparsity_schedule) else (int)(num_train_steps / len(sparsity_schedule) / 5)
          finetune_step = 3000 if pruning_step == 100 else pruning_step*4
          for s in sparsity_schedule:
            pruning_hook.initialize(s, pruning_step, 1)
            pruning_estimator.train(input_fn=train_input_fn, steps=pruning_step, hooks=[pruning_hook])
            print_evaluating_result(estimator, eval_input_fn, eval_steps)
            estimator.train(input_fn=train_input_fn, steps=finetune_step)
            print_evaluating_result(estimator, eval_input_fn, eval_steps)
            have_trained_step += (pruning_step+finetune_step)
        if (num_train_steps > have_trained_step):
          estimator.train(input_fn=train_input_fn, steps=num_train_steps-have_trained_step)
    elif FLAGS.mode == "pretrain":
      estimator.train(input_fn=train_input_fn, steps=100)#for check
      estimator.train(input_fn=train_input_fn, steps=num_train_steps)
    print_evaluating_result(estimator, eval_input_fn, eval_steps)
  



if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("sparsity")
  tf.app.run()
