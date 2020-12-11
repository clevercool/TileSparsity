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
import math
import os
import sys
import pickle
# import _pickle as pickle
import numpy as np
import tensorflow as tf

def pause():
    sys.stdout.flush()
    sys.stdin.read(1)

class logPerformanceHook(tf.train.SessionRunHook):
    def __init__(self, logits, label_ids, is_real_example, per_example_loss, total_num_steps):
      predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      self.accuracy = tf.metrics.accuracy(
        labels=label_ids, predictions=predictions, weights=is_real_example)
      self.loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
      self.iteration = 0
      self.total_num_steps = total_num_steps
      self.log_step = 500 if ((int)(self.total_num_steps/50)) > 100 else 100


    def begin(self):
        self.iteration = 0
      
    def after_run(self, run_context, run_values):
      run_context.session.run([self.loss[0], self.accuracy[0]])
      self.iteration += 1

      if self.iteration > 0 and self.iteration % self.log_step == 0:
        self.printRecord(run_context.session)

    def end(self, session):
      self.printRecord(session)

    def printRecord(self, session):
      average_loss, average_accuray = session.run([self.loss[1], self.accuracy[1]])
      tf.logging.info("[INFO] Iteration %d: training loss: %f, training accuracy: %f", self.iteration, average_loss, average_accuray)

class exportPrunedModel(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        all_variables = tf.global_variables()
        ckpt = {}
        for i in range(len(all_variables)):
            if all_variables[i].name.find("mask-i") != -1:
                mi = all_variables[i]
                mo = all_variables[i+1]
                kernel = all_variables[i+2]
                bias = all_variables[i+3]
                mi_val, mo_val, kernel_val, bias_val = session.run([mi, mo, kernel, bias])

                def convert_mask_to_offset(mask):
                    offset = []
                    index = 0
                    for i in range(len(mask)):
                        if mask[i] == 1:
                            offset.append(index)
                            offset[-1] = offset[-1] - (len(offset) - 1)
                        index += 1
                    return np.array(offset)

                mi_offset = convert_mask_to_offset(mi_val)
                mo_offset = convert_mask_to_offset(mo_val)
                
                bias_val = bias_val*mo_val
                kernel_val = kernel_val[np.array(mi_val, dtype=bool)]
                kernel_val = (kernel_val.T[np.array(mo_val, dtype=bool)]).T
                ckpt[mi.name] = mi_offset
                ckpt[mo.name] = mo_offset
                ckpt[kernel.name] = kernel_val
                ckpt[bias.name] = bias_val
                i = i + 4

        with open('pruned_model.pkl', 'wb') as f:
          pickle.dump(ckpt, f, pickle.HIGHEST_PROTOCOL)

class InitializeGlobalStepHook(tf.train.SessionRunHook):
    def begin(self):
      self.initial_global_step = tf.assign(tf.train.get_or_create_global_step(), 0)

    def after_create_session(self, session, coord):
      session.run(self.initial_global_step)

class TaylorRanker():
  def __init__(self):
    pass

  def get_var(self):
    ### collect the outputs of masked layers and loss layer
    loss = None
    masked_output = []
    for op in tf.get_default_graph().get_operations():
      for t in op.values():
        if t.name.find("mask-output") != -1 or t.name.find("mask-input") != -1:
          masked_output.append(t)
        elif t.name.find("loss/Mean:0") != -1:
          loss = t
    return masked_output, loss

  def compute_score(self, mask_var):
    masked_output, loss = self.get_var()
    var_grad = tf.gradients(loss, mask_var)
    scores = [self.square_score(v, m) for v, m in zip(var_grad, masked_output)]
    return scores

  # BY "Importance Estimation for Neural Network Pruning"
  # The latest paper of NV
  def square_score(self, var_grad, masked_output):
    # find that sum of square is better than square of sum
    # score = tf.math.square(tf.reduce_sum(var_grad*masked_output, axis=0)) # square of sum
    score = tf.reduce_sum(tf.math.square(var_grad*masked_output), axis=0) # sum of square 
    return score

  # BY "Pruning Convolutional Neural Networks for Resource Efficient Inference"
  # Older taylor approximation method
  def abs_mean_score_with_l2norm(self, var_grad, masked_output):
    score = tf.math.abs(tf.reduce_mean(var_grad*masked_output, axis=0))
    norm = tf.math.sqrt(tf.reduce_sum(score*score))
    if norm != 0.0:
        score = score / norm
    return score

  # BY myself, wrongly implement the NV paper,
  # but the performance seems be better than abs_mean_score_with_l2_norm
  # need to check
  def abs_sum_score_with_l1norm(self, var_grad, masked_output):
    score = tf.math.abs(tf.reduce_sum(var_grad*masked_output, axis=0))
    norm = tf.reduce_sum(score*score)
    if norm != 0.0:
        score = score / norm
    return score    

  def pruning(self, accumulated_scores, sparsity, placeholders, sess, assign_ops):
    ### pruning
    # s = [y for x in accumulated_scores for y in x]
    threshold = np.percentile([y for x in accumulated_scores for y in x], sparsity)
    mask = [ a_score > threshold for a_score in accumulated_scores ]
    feed = dict([ (p, m) for p, m in zip(placeholders, mask)])
    # assign the updated mask 
    sess.run(assign_ops, feed)

class SparseColumnPruningRank():

  def __init__(self, ):
    pass

  def compute_score(self, mask_var):
    return self.compute_taylor_score(mask_var)
    # return self.comput_weight_score(mask_var)

  def compute_weight_score(self, mask_var):
    kernel_value = self.get_weight_var()
    return kernel_value

  def get_weight_var(self):
    kernel_value = []
    ### collect the outputs of masked layers and loss layer
    for op in tf.get_default_graph().get_operations():
      for t in op.values():
        if t.name.find("mask-kernel") != -1:
          kernel_value.append(t)
    return kernel_value

  def compute_taylor_score(self, mask_var):
    masked_output, loss = self.get_taylor_var()
    var_grad = tf.gradients(loss, mask_var)
    scores = [ tf.math.square(v * m) for v, m in zip(var_grad, masked_output)]
    return scores

  def get_taylor_var(self):
    ### collect the outputs of masked layers and loss layer
    loss = None
    masked_output = []
    
    #[print(t) for op in tf.get_default_graph().get_operations() for t in op.values()]
    #[print(t.name) for op in tf.get_default_graph().get_operations() for t in op.values()]
    #[print(t.name.find("/weights:0")) for op in tf.get_default_graph().get_operations() for t in op.values()]
    #[print(t.name.find("loss/Mean:0")) for op in tf.get_default_graph().get_operations() for t in op.values()]
    
    for op in tf.get_default_graph().get_operations():
      for t in op.values():
        if t.name.find("mask-kernel") != -1:
          masked_output.append(t)
        elif t.name.find("loss/Mean:0") != -1:
          loss = t
    return masked_output, loss

  def granularity_pruning(self, hidden_units, granularity, score, sparsity):
    block_num = score.shape[1] / granularity
    split_score = np.split(score, block_num, axis=1)
    split_l2_norm = [ np.linalg.norm(w, axis=1) for w in split_score ]
    split_mask = []
    for normed_score in split_l2_norm:
        threshold = np.percentile(normed_score, sparsity)
        split_mask.append(normed_score > threshold)
#     threshold = np.percentile([ v for n in split_l2_norm for v in n], sparsity)
#     split_mask = [ norm > threshold for norm in split_l2_norm ]
    split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(granularity)], axis=1) for m in split_mask ]
    mask = np.concatenate(split_mask, axis=1)
    return mask

  def pruning(self, task_name, accumulated_scores, sparsity, placeholders, sess, assign_ops, pruning_type, granularity, block_remain = 0, skip_block_dim = 0, g1percent = 0):
    head_num = 12
    size_per_head = 64
    hidden_units = head_num * size_per_head
    feed = {}

    if hidden_units % granularity != 0:
      print("[ERROR] hidden_units %d cannot be divided by granularity %d" % (hidden_units, granularity))
      exit(-1)
          
    if pruning_type == 7:
      # TW pruning on dim K.
      # granularity pruning on whole layer
      l2_norm_list = []
      transpose = False
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        if transpose:
          score = score.T
        block_num = score.shape[1] / granularity
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        l2_norm_list.append(split_l2_norm)
      
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]
      threshold = np.percentile(split_weight_l2_norm_list, sparsity)

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        if transpose:
          score = score.T
        block_num = score.shape[1] / granularity
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(granularity)], axis=1) for m in split_mask ]
        mask = np.concatenate(split_mask, axis = 1)
        if transpose:
          mask = mask.T
        assert(mask.shape == placeholders[layer].shape)
        feed[placeholders[layer]] = mask

      sess.run(assign_ops, feed)
      return

    if pruning_type == 8:
      # TW prune N + K dimension on whole network
      # N dimension pruning whole column
      # K dimension pruning uses granularity pruning
      # prune N dimesnion first is necessary, and hence consider the pruning on K and N independently. 
      
      # pruning on N
      l2_norm_list = [ np.linalg.norm(score, axis=0) / score.shape[0] for score in accumulated_scores ]
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      threshold = np.percentile(split_layer_l2_norm_list, sparsity)
      mask = [ norm > threshold for norm in l2_norm_list ]
      tiled_mask = [ np.tile(m, (score.shape[0], 1)).reshape(score.shape) for m, score in zip(mask, accumulated_scores)]
      pruned_accumulated_scores = [ m * score for m, score in zip(tiled_mask, accumulated_scores) ]
      
      # pruning on K
      l2_norm_list = []
      split_score_list = []
      for layer in range(len(pruned_accumulated_scores)):
          score = pruned_accumulated_scores[layer]
          split_point = []
          column_number = 0
          column_number_list = []
          for column_index in range(score.shape[1]):
            if mask[layer][column_index] == 1:
                column_number += 1
            if column_number == granularity:
                if column_index + 1 < score.shape[1]:
                    split_point.append(column_index + 1)
                    column_number_list.append(column_number)
                    column_number = 0
          column_number_list.append(column_number)
          split_score = np.split(score, split_point, axis=1)
          split_score_list.append(split_score)
          split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for i, w in enumerate(split_score) ]
          l2_norm_list.append(split_l2_norm)

      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_vector_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]
      threshold = np.percentile(split_vector_l2_norm_list, sparsity)

      for layer in range(len(pruned_accumulated_scores)):
        score = pruned_accumulated_scores[layer]
        split_score = split_score_list[layer]
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ mask.reshape(mask.shape[0], 1) for mask in split_mask ]
        split_mask = [ np.tile(m, (1, split_score[i].shape[1])).reshape(split_score[i].shape) for i, m in enumerate(split_mask) ]
        mask = np.concatenate(split_mask, axis = 1)
        
        assert(mask.shape == tiled_mask[layer].shape)
        assert(tiled_mask[layer].shape == placeholders[layer].shape)
        feed[placeholders[layer]] = mask * tiled_mask[layer]

      sess.run(assign_ops, feed)
      return

           

    layer = 0
    while layer < len(accumulated_scores):
      if layer % 6 < 3:
        # layer 0, 1, 2 are Q, K, V
        if pruning_type == 0 or pruning_type == 1:
          # pruning_type 0: prune head_number, like prune from 12 heads to 6 heads
          # pruning_type 1: prune size per head, like prune from 64 to 32.
          score = accumulated_scores[layer]
          score = np.reshape(score, [head_num, size_per_head, hidden_units])
          if pruning_type == 1:
              score = np.transpose(score, [1, 0, 2])
          score_shape = score.shape
          score = np.reshape(score, [score_shape[0], score_shape[1]*score_shape[2]])

          l2_norm_score = np.linalg.norm(score, axis=1)
          threshold = np.percentile(l2_norm_score, sparsity)
          mask = l2_norm_score > threshold
          mask = np.expand_dims(mask, axis=1)
          mask = np.concatenate([mask for i in range(score_shape[1] * score_shape[2])], axis=1)
          mask = np.reshape(mask, score_shape)
          if pruning_type == 1:
            mask = np.transpose(mask, [1, 0, 2])
          mask = np.reshape(mask, [hidden_units, hidden_units])
          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1
        elif pruning_type == 2:
          # granularity-prune 
          score = accumulated_scores[layer]
          mask = self.granularity_pruning(hidden_units, granularity, score, sparsity)
          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1
        elif pruning_type == 3:
          score = np.concatenate([accumulated_scores[layer], accumulated_scores[layer+1], accumulated_scores[layer+2]], axis=1)
          mask = self.granularity_pruning(hidden_units*3, hidden_units*3, score, sparsity)
          mask_list = np.split(mask, 3, axis=1)
          for i in range(3):
            assert(placeholders[layer + i].shape == mask_list[i].shape)
            feed[placeholders[layer + i]] = mask_list[i]
          layer += 3
        elif pruning_type == 4:
          score = accumulated_scores[layer]
          transpose = True # True when pruning N dim, False when pruning K dim
          if transpose == True:
              score = score.T
          
          block_size = 16
          # topk = 4
          # assert(sparsity == int(100 * (1 - topk / block_size)))
          topk = int((1-sparsity/100)*block_size)
          mask = np.zeros(score.shape)
          for i in range(score.shape[0]):
              j = 0
              while j < score.shape[1]:
                  threshold = np.sort(score[i][j:j+block_size])[-topk]
                  mask[i][j:j+block_size] = score[i][j:j+block_size] >= threshold
                  j += block_size
          if transpose == True:
            mask = mask.T
          
          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1
        
        elif pruning_type == 5:
          score = accumulated_scores[layer]
          block_size = granularity
          norm_list = []
          i = 0
          while i < score.shape[0]:
              j = 0
              while j < score.shape[1]:
                  l2_norm_value = np.linalg.norm(score[i:i+block_size, j:j+block_size])
                  norm_list.append(l2_norm_value)
                  j += block_size
              i += block_size

          threshold = np.percentile(norm_list, sparsity)
          block_mask = norm_list > threshold
          mask = np.zeros(score.shape)
          mask_index = 0
          i = 0
          while i < mask.shape[0]:
              j = 0
              while j < mask.shape[1]:
                  if block_mask[mask_index] == True:
                      mask[i:i+block_size, j:j+block_size] = 1
                  mask_index += 1
                  j += block_size
              i += block_size

          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1
        elif pruning_type == 6:
          # granularity-prune on K dimension
          # prune whole column for N dimension
          score = accumulated_scores[layer]
          #mask = self.granularity_pruning(hidden_units, granularity, score, sparsity)
          mask = self.granularity_pruning(hidden_units, 32, score, sparsity)
          transpose_score = score.T
          transpose_mask = self.granularity_pruning(hidden_units, transpose_score.shape[1], transpose_score, sparsity)
          and_mask = mask & transpose_mask.T
          assert(placeholders[layer].shape == and_mask.shape)
          feed[placeholders[layer]] = and_mask
          layer += 1
        else:
          print("[ERROR] pruning type must be one of 0, 1, 2, 3, 4, 5.")
          exit(-1)
      else:
        # layer 4, 5, 6 are ffn, use granularity prune directly.
        if pruning_type == 6:
          # granularity-prune on K dimension
          # prune whole column for N dimension
          score = accumulated_scores[layer]
          #mask = self.granularity_pruning(hidden_units, granularity, score, sparsity)
          mask = self.granularity_pruning(hidden_units, 128, score, sparsity)
          transpose_score = score.T
          transpose_mask = self.granularity_pruning(hidden_units, transpose_score.shape[1], transpose_score, sparsity)
          and_mask = mask & transpose_mask.T
          assert(placeholders[layer].shape == and_mask.shape)
          feed[placeholders[layer]] = and_mask
          layer += 1
        elif pruning_type == 4:
          score = accumulated_scores[layer]
          transpose = True # True when pruning N dim, False when pruning K dim
          if transpose == True:
              score = score.T
          
          block_size = 16
          # topk = 4
          # assert(sparsity == int(100 * (1 - topk / block_size)))
          topk = int((1-sparsity/100)*block_size)
          mask = np.zeros(score.shape)
          for i in range(score.shape[0]):
              j = 0
              while j < score.shape[1]:
                  threshold = np.sort(score[i][j:j+block_size])[-topk]
                  mask[i][j:j+block_size] = score[i][j:j+block_size] >= threshold
                  j += block_size
          if transpose == True:
            mask = mask.T
          
          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1       
        elif pruning_type == 5:
          score = accumulated_scores[layer]
          block_size = granularity
          norm_list = []
          i = 0
          while i < score.shape[0]:
              j = 0
              while j < score.shape[1]:
                  l2_norm_value = np.linalg.norm(score[i:i+block_size, j:j+block_size])
                  norm_list.append(l2_norm_value)
                  j += block_size
              i += block_size

          threshold = np.percentile(norm_list, sparsity)
          block_mask = norm_list > threshold
          mask = np.zeros(score.shape)
          mask_index = 0
          i = 0
          while i < mask.shape[0]:
              j = 0
              while j < mask.shape[1]:
                  if block_mask[mask_index] == True:
                      mask[i:i+block_size, j:j+block_size] = 1
                  mask_index += 1
                  j += block_size
              i += block_size

          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1
        else:
          score = accumulated_scores[layer]
          mask = self.granularity_pruning(score.shape[1], granularity, score, sparsity)
          assert(placeholders[layer].shape == mask.shape)
          feed[placeholders[layer]] = mask
          layer += 1

    sess.run(assign_ops, feed)

class NetworkPruningHook(tf.train.SessionRunHook):
    def __init__(self, flags):
      super(NetworkPruningHook, self).__init__()
      self.sparsity = 0
      self.total_step = 0
      self.step = 0
      self.hasInitialize = False
      self.flags=flags
      self.momentum = 0.9

    def initialize(self, sparsity, total_step, total_stage=1):
      self.step = 0
      self.total_step = (int)(total_step)
      self.sparsity = [ (int)(sparsity*(i+1)/total_stage) for i in range(total_stage) ]
      self.pruning_time = [ (int)(total_step*(i+1)/total_stage) for i in range(total_stage) ]
      self.hasInitialize = True

    def begin(self):
      tf.logging.info("[INFO] sparsity: %d with %d stages", self.sparsity[-1], len(self.sparsity))
      self.mask_var = []
      self.assign_ops = []
      self.placeholders = []
      self.accumulated_scores = []
      self.saver = tf.train.Saver(tf.global_variables())
      self.global_step = tf.train.get_or_create_global_step()
      assert(self.hasInitialize == True)

      ### collect the masked variables
      for v in tf.global_variables():
        if v.name.find("mask-i:0") != -1 or v.name.find("mask-o:0") != -1:
          self.mask_var.append(v)
          p = tf.placeholder(tf.float32, shape=v.shape)
          self.placeholders.append(p)
          self.assign_ops.append(tf.assign(v, p))
          self.accumulated_scores.append(np.zeros(v.shape))
      sys.stdout.flush()

      self.ranker = SparseColumnPruningRank()
      self.compute_score = self.ranker.compute_score(self.mask_var)

    def before_run(self, run_context):
      if self.step > self.pruning_time[-1]:
        return
      return tf.train.SessionRunArgs(self.compute_score)

    def after_run(self, run_context, run_values):
      if self.step > self.pruning_time[-1]:
        return
      for i in range(len(self.accumulated_scores)):
        self.accumulated_scores[i] += run_values.results[i]
        # self.accumulated_scores[i] = self.accumulated_scores[i]*self.momentum + run_values.results[i]*(1-self.momentum)
      self.step += 1
      if self.step in self.pruning_time:
        sparsity = self.sparsity[self.pruning_time.index(self.step)]
        
        ### check the zero_density
        mask_val_before = run_context.session.run(self.mask_var)
        average = np.average([y for mask_each_layer in mask_val_before for y in mask_each_layer.flatten()])
        tf.logging.info("The average zero_density of network before pruning: %f", average)

        # ### pruning
        self.ranker.pruning(self.flags.task_name, self.accumulated_scores, sparsity, self.placeholders, run_context.session, self.assign_ops, self.flags.pruning_type, self.flags.granularity, self.flags.block_remain, self.flags.skip_block_dim, self.flags.g1percent)

        ### check the zero_density
        mask_val_after = run_context.session.run(self.mask_var)
        average = np.average([y for mask_each_layer in mask_val_after for y in mask_each_layer.flatten()])
        tf.logging.info("The average zero_density of network after pruning: %f", average)

        # show the pruning ratio of each layer
        sum_before_pruning_this_stage = 0.
        sum_after_pruning_this_stage = 0.
        total_FLOPs_before_pruning = 0.
        tf.logging.info("FLOP reduction ratio of each dense layer: ")
        for i in range(0, len(self.mask_var)):
            before_o = np.sum(mask_val_before[i], dtype=np.int32)
            after_o = np.sum(mask_val_after[i], dtype=np.int32)
            tf.logging.info("layer %3d, {:55}: FLOPs: %7d -> %7d (speedup: %1.3f)".format(self.mask_var[i].name), 
                    i, before_o, after_o, before_o * 1.0 / after_o)
            sum_before_pruning_this_stage += before_o
            sum_after_pruning_this_stage += after_o
            total_FLOPs_before_pruning += np.sum(mask_val_before[i])


        average_FLOP_reduction_this_stage = 100*(1-(sum_after_pruning_this_stage/sum_before_pruning_this_stage))
        total_FLOP_reduction = 100*(1-(sum_after_pruning_this_stage/total_FLOPs_before_pruning))
        tf.logging.info("[INFO] sparsity: %d, Average FLOP reduction: %2.3f%%, Total FLOP reduction: %2.3f%%",
         sparsity, average_FLOP_reduction_this_stage, total_FLOP_reduction)

        output_eval_file = os.path.join(self.flags.output_dir, "Important_result.txt")
        with tf.gfile.GFile(output_eval_file, "a") as writer:
          writer.write("[INFO] sparsity: %d, Average FLOP reduction: %2.3f%%, Total FLOP reduction: %2.3f%% \n" %
             (sparsity, average_FLOP_reduction_this_stage, total_FLOP_reduction))

        for i in range(len(self.accumulated_scores)):
            self.accumulated_scores[i] = np.zeros(self.accumulated_scores[i].shape)

    def end(self, session):
        path = "%s/model.ckpt-%d" % (self.flags.output_dir, session.run(self.global_step))
        tf.logging.info("[INFO] saved path: %s", path)
        self.saver.save(session, path)

