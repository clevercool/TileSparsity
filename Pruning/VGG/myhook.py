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
import numpy as np
import tensorflow as tf
import prune_algo

class SparseColumnPruningRank():

    def __init__(self, ):
        pass

    def compute_score(self, mask_var):
        return self.compute_taylor_score(mask_var)
        #return self.compute_weight_score(mask_var)

    def compute_weight_score(self, mask_var):
        kernel_value = self.get_weight_var()
        return kernel_value

    def get_weight_var(self):
        kernel_value = []
        ### collect the outputs of masked layers and loss layer
        for op in tf.get_default_graph().get_operations():
            for t in op.values():
                if t.name.find("/weights:0") != -1:
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
        #[print(t.name) for op in tf.get_default_graph().get_operations() for t in op.values()]
        #[print(n.values()) for n in tf.get_default_graph().get_operations()]
        for op in tf.get_default_graph().get_operations():
            for t in op.values():
                if t.name.find("/weights:0") != -1:
                    masked_output.append(t)
                elif t.name.find("loss/Sum:0") != -1:
                    loss = t
                    
        return masked_output, loss


class PruningHook(tf.train.SessionRunHook):
    def __init__(self, masks, pruning_type, is_masked_weight, mask_values={}):
        super().__init__()
        self.masks = masks
        self.pruning_type = pruning_type
        self.is_masked_weight = is_masked_weight
        self.masks_now = None
        self.mask_values = mask_values
        self.mask_placeholders = None
        self.masked_weights = None
        self.is_pruning = False
        self.pruning_steps = 0

        ######################################
        self.sparsity = 0
        self.step = 0
        self.hasInitialize = False
        #self.flags=flags
        self.momentum = 0.9
        ######################################

    def initialize(self, pruning_steps):
        self.step = 0
        self.pruning_steps=pruning_steps

    def begin(self):
        #tf.logging.info("[INFO] sparsity: %d with %d stages", self.sparsity[-1], len(self.sparsity))
        self.mask_var = []
        self.assign_ops = []
        self.placeholders = []
        self.accumulated_scores = []
        #self.saver = tf.train.Saver(tf.global_variables())
        #self.global_step = tf.train.get_or_create_global_step()

        if not self.is_pruning:
            return

        #if len(self.mask_values) == 0:
        for name, tensor in self.mask_placeholders.items():
            if name not in self.mask_values.keys():
                self.mask_values[name] = np.ones(tensor.shape, dtype=np.int)
                
        ### collect the masked variables
        ####################################
        for name, tensor in self.mask_placeholders.items():
            self.mask_var.append(tensor)
            self.accumulated_scores.append(np.zeros(tensor.shape))
        sys.stdout.flush()
        ####################################

        if not self.is_masked_weight:
            self.ranker = SparseColumnPruningRank()
            self.compute_score = self.ranker.compute_score(self.mask_var)

    def before_run(self, run_context):
        if self.is_pruning and not self.is_masked_weight:
            return tf.train.SessionRunArgs(
                fetches=[self.masked_weights, self.compute_score],
                feed_dict={
                    placeholder: self.mask_values[name]
                    for name, placeholder in self.mask_placeholders.items()
                },
            )
        
        if self.is_pruning:
            return tf.train.SessionRunArgs(
                fetches=[self.masked_weights],
                feed_dict={
                    placeholder: self.mask_values[name]
                    for name, placeholder in self.mask_placeholders.items()
                },
            )

        return tf.train.SessionRunArgs()

    def print_weights(self):
        for weight in tf.trainable_variables():
            if "kernel" in weight.name:
                print("\"%s\"," %weight.name)

    def insert_masks(self):
        if not self.is_pruning:
            assert(self.masks)
            assert(self.mask_values)
            for weight in tf.trainable_variables():
                name = weight.name
                if name in self.masks:
                    read_op = next(
                        op for op in weight.op.outputs[0].consumers() if op.type == "Identity"
                    )
                    tensor = read_op.outputs[0]
                    downstream_ops = tensor.consumers()
                    ### Make the mask_values has the same number with masks
                    assert(self.mask_values[name].shape == tensor.shape)
                    mask_constant = tf.constant(self.mask_values[name], dtype=tf.float32)
                    masked_weight = tensor * mask_constant
                    for downstream_op in downstream_ops:
                        downstream_op._update_input(
                            list(downstream_op.inputs).index(tensor), masked_weight
                        )
            self.mask_placeholders = None
            self.masked_weights = None
        else:
            mask_placeholders = {}
            masked_weights = {}
            for weight in tf.trainable_variables():
                name = weight.name
                if name in self.masks:
                    read_op = next(
                        op for op in weight.op.outputs[0].consumers() if op.type == "Identity"
                    )
                    tensor = read_op.outputs[0]
                    downstream_ops = tensor.consumers()
                    mask_placeholder = tf.placeholder(dtype=tf.float32, shape=tensor.shape)
                    masked_weight = tensor * tf.cast(mask_placeholder, dtype=tf.float32)
                    for downstream_op in downstream_ops:
                        downstream_op._update_input(
                            list(downstream_op.inputs).index(tensor), masked_weight
                        )
                    mask_placeholders[name] = mask_placeholder
                    masked_weights[name] = masked_weight
            
            self.mask_placeholders = mask_placeholders
            self.masked_weights = masked_weights

    def prune(self, masks_name, masked_weights, mask_values, accumulated_scores, is_masked_weight, masks_now):
        # Prune

        scores = []
        #masked_weights or taylor
        if is_masked_weight:
            for layer in masks_now:
                scores.append(masked_weights[masks_name[layer]])
        else:
            for layer in masks_now:
                scores.append(accumulated_scores[layer])

        # Img2col
        # the new_accumulated_scores are positive numbers.
        new_accumulated_scores = prune_algo.img2col_forward(scores)
		
        new_mask_values = prune_algo.pruning_fun(self.pruning_type)(new_accumulated_scores, self.sparsity)

        new_mask_values = prune_algo.img2col_back_ward(new_mask_values, scores)


        num_1 = 0
        num_t = 0
        new_mask_values_dict = mask_values

        for layer in range(len(masks_now)):
            new_mask_values_dict[masks_name[masks_now[layer]]] = new_mask_values[layer]
        
        for layer in range(len(new_mask_values_dict)):
            mask = new_mask_values_dict[masks_name[layer]]
            a = np.array(mask.flatten().tolist())
            sa = np.sum(a)
            tf.logging.info("  Layer : %d %s" %(layer, masks_name[layer]))
            tf.logging.info("    Shape   :")
            tf.logging.info((mask.shape))
            tf.logging.info("    0       : %d" %(len(a)-sa))
            tf.logging.info("    1       : %d" %(sa))
            tf.logging.info("    Length  : %d" %(len(a)))
            tf.logging.info("    Density : %f" %(sa/len(a)))
            num_1 += sa
            num_t += len(a)
        
        tf.logging.info("\n\nALL Density : %f \n" %(num_1/num_t))

        return new_mask_values_dict

    def after_run(self, run_context, run_values):
        if not self.is_pruning:
            return

        self.step += 1

        if self.step != self.pruning_steps:
            return

        masked_weights = run_values.results[0]

        if not self.is_masked_weight:
            compute_score = run_values.results[1]
            for i in range(len(self.accumulated_scores)):
                self.accumulated_scores[i] += compute_score[i]

        self.mask_values = self.prune(self.masks, masked_weights, self.mask_values, self.accumulated_scores, self.is_masked_weight, self.masks_now)

        if not self.is_masked_weight:
            for i in range(len(self.accumulated_scores)):
                self.accumulated_scores[i] = np.zeros(self.accumulated_scores[i].shape)

    def end(self, session):
        pass
  