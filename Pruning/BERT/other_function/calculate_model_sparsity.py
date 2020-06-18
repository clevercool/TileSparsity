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

#!/usr/bin/env python3
import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import pickle

sys.path.append('./utils')
import modeling as  modeling

if __name__=="__main__":
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--ckpt_dir", default=None, type=str, 
                      help="model ckpt dir. E.g., MNLI/model_0/model.ckpt-24643")
  parser.add_argument("--output_dir", default='./profile/tw2d_19_mix_1_56%__prune_density_G128.pickle', type=str, 
                      help="output sparsity pickle file dir")  
  args = parser.parse_args()
  

  
  
  bert_config = modeling.BertConfig.from_json_file('../Model/uncased_L-12_H-768_A-12/bert_config.json')

  input_ids = tf.placeholder(tf.int32,(8,256))

  model = modeling.BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ckpt_dir = args.ckpt_dir

    saver.restore(sess, ckpt_dir)

    #print(f'parsing file {ckpt_dir}')
    
    results = []
    for probe_layer in range(0,12):
    
      mask_wq = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/query/mask-o:0')
      mask_wk = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/key/mask-o:0')
      mask_wv = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/value/mask-o:0')
      mask_fc1 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/output/dense/mask-o:0')
      mask_fc2 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/intermediate/dense/mask-o:0')
      mask_fc3 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/output/dense/mask-o:0')    
      
      result = sess.run([mask_wq,mask_wk,mask_wv,mask_fc1,mask_fc2,mask_fc3])
      results.append(result)

  prune_mask = np.asarray(results)
  prune_density_list = []

  with open("./prune_mask/tw2d_8_46%__prune_density_G128.pickle.pkl", mode='wb') as f:
    pickle.dump(prune_mask, f, protocol=3)
  for layer_idx in range(prune_mask.shape[0]):
    layer_list = []
    for matrix_idx in range(prune_mask.shape[1]):
      # remain 1, prune 0
      #split_mask = np.split(prune_mask[layer_idx,matrix_idx],1,axis=1)
      split_mask = np.split(prune_mask[layer_idx,matrix_idx],prune_mask[layer_idx,matrix_idx].shape[1]/64,axis=1)
      prune_density = [1-np.sum(split)/(split.shape[0]*split.shape[1]) for split in split_mask]
      layer_list.append(prune_density)
    prune_density_list.append(layer_list)  
  # print(prune_density_list)
  #print(f'dumping sparsity to: {args.output_dir}')
  if not os.path.exists(os.path.dirname(args.output_dir)):
    os.makedirs(os.path.dirname(args.output_dir))
  with open(args.output_dir, mode='wb') as f:
    pickle.dump(prune_density_list, f, protocol=3)

  with open(args.output_dir, "rb") as f:
    data = pickle.load(f)
    for layer in data:
        layer[5] = [i * 4 for i in layer[5]]
    ele = [w for m in data for n in m for w in n]
    sum = np.sum(ele)
    #print("Sparsity")
    print(sum/1728)
  
  #np.savetxt(args.output_dir,np.asarray(prune_density_list).reshape((12*6,1)),delimiter=',')
