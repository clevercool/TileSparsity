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
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import math
import six
from datetime import datetime
import sys
import time
import pickle 

if len(sys.argv) != 3:
    print("[ERROR] python dump_pruned_model.py input_file_name, output_file_name")
    print("[ERROR] e.g. python model-73311.ckpt 50-masked-model")
    sys.exit(0)

ckpt_name = sys.argv[1]
output_filename = sys.argv[2]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckpt_name + ".meta")
    saver.restore(sess, (ckpt_name))

    def dumpModel():
        total_variables = np.array([])
        pruned_variables = np.array([])

        all_variables = tf.global_variables()
        origin_ckpt = {} # original model, no mask
        masked_ckpt = {} # set the reduant rows and columns of weights to zero, but no remove
        pruned_ckpt = {} # remove the redudant rows and colums of weights
        other_variable_ckpt = {} # only store the variables which are not in the transformer
        absolute_mask = {} # mask with boolean type, [0 1 0 0 1 1 0 1]
        relative_mask = {} # mask with relative offset, [0 1 1 1 2 3 3 4]
        layer_norm_ckpt = {}
        model_information = {}
        for i in range(len(all_variables)):
            print("[INFO] %d/%d %s" %(i, len(all_variables), all_variables[i].name), '\r')
            sys.stdout.flush()
            if all_variables[i].name.find("mask-o") != -1:
                # mi = all_variables[i]
                kernel = all_variables[i-2]
                bias = all_variables[i-1]
                mo = all_variables[i]
                mo_val, kernel_val, bias_val = sess.run([mo, kernel, bias])

                total_variables = np.concatenate((total_variables.flatten(), kernel_val.flatten(), bias_val.flatten()), axis = 0)

                def convert_mask_to_offset(mask):
                    return mask
                    # offset = []
                    # index = 0
                    # for i in range(len(mask)):
                    #     if mask[i] == 1:
                    #         offset.append(index)
                    #         offset[-1] = offset[-1] - (len(offset) - 1)
                    #     index += 1
                    # return np.array(offset)

                origin_ckpt[kernel.name] = kernel_val
                origin_ckpt[bias.name] = bias_val

                # absolute_mask[mi.name] = mi_val
                absolute_mask[mo.name] = mo_val

                masked_kernel_val = kernel_val * mo_val
                # masked_kernel_val = masked_kernel_val.T * mi_val
                # masked_kernel_val = masked_kernel_val.T

                # masked_bias_val = bias_val * mo_val

                masked_ckpt[kernel.name] = masked_kernel_val
                # masked_ckpt[bias.name] = masked_bias_val

                # mi_offset = convert_mask_to_offset(mi_val)
                # mo_offset = convert_mask_to_offset(mo_val)

                # relative_mask[mi.name] = mi_offset
                # relative_mask[mo.name] = mo_offset
                
                # pruned_kernel_val = kernel_val[np.array(mi_val, dtype=bool)]
                # pruned_kernel_val = (pruned_kernel_val.T[np.array(mo_val, dtype=bool)]).T

                # pruned_ckpt[kernel.name] = pruned_kernel_val
                # pruned_ckpt[bias.name] = masked_bias_val # we do not remove the redundant bias 

                # pruned_variables = np.concatenate((pruned_variables.flatten(), pruned_kernel_val.flatten(), masked_bias_val.flatten()), axis=0)

                # print("[INFO] kernel info: ", kernel.name, "[%d] -> [%d, %d], %f" %(len(mi_val), len(mo_val), len(mi_offset),
                #      len(mo_offset), 1.0 * len(mi_val) * len(mo_val) / len(mi_offset) / len(mo_offset)))
                
                # info = kernel.name + " [%d, %d] -> [%d, %d], %f \n" % (len(mi_val), len(mo_val), len(mi_offset),
                #     len(mo_offset), 1.0 * len(mi_val) * len(mo_val) / len(mi_offset) / len(mo_offset) )
                # model_information[kernel.name] = info
                i = i + 3
            else:
                var = all_variables[i]
                if var in tf.trainable_variables():
                    if (var.name not in masked_ckpt) and (var.name not in absolute_mask):
                        val = sess.run(var)
                        if var.name.find("LayerNorm") != -1:
                            layer_norm_ckpt[var.name] = val
                        else:
                            other_variable_ckpt[var.name] = val
        model = {}
        model["transformer_ckpt"] = {}
        model["transformer_ckpt"]["origin_ckpt"] = origin_ckpt
        model["transformer_ckpt"]["masked_ckpt"] = masked_ckpt
        # model["transformer_ckpt"]["pruned_ckpt"] = pruned_ckpt
        model["transformer_ckpt"]["absolute_mask"] = absolute_mask
        # model["transformer_ckpt"]["relative_mask"] = relative_mask
        # model["transformer_ckpt"]["layer_norm"] = layer_norm_ckpt
        # model["other_variable_ckpt"] = other_variable_ckpt
        # model["information"] = model_information
        
        # origin_mean = np.mean(total_variables)
        # origin_std = np.std(total_variables)
        # pruned_mean = np.mean(pruned_variables)
        # pruned_std = np.std(pruned_variables)

        # summary = "Original model: \n    Total variables: %d \n    mean: %f \n    std: %f\n" % (len(total_variables), origin_mean, origin_std)
        # summary += "Pruned model: \n    Total variables: %d \n    mean: %f \n    std: %f\n" % (len(pruned_variables), pruned_mean, pruned_std)
        # summary += "Total compression rate: %f\n" % (len(total_variables) * 1.0 / len(pruned_variables))
        # model["summary"] = summary

        with open(output_filename + ".pkl", "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    dumpModel()
