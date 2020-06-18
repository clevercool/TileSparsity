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
import matplotlib.pyplot as plt


sys.path.append('./utils')
import modeling as  modeling

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt_dir", default='/home/chandler/mask-bert/model/ew_75/model.ckpt-24643', type=str, 
                      help="model ckpt dir. E.g., MNLI/model_0/model.ckpt-24643")
  parser.add_argument("--output_dir", default='./fig/masked_weight.png', type=str, 
                      help="output png file dir")
  parser.add_argument("--layer_index", default=0, type=int, 
                      help="layer to plot")  
  parser.add_argument("--matrix_name", default="wq", type=str, 
                      help="name of the matrix to plot. (wq, wk, wv, fc1, fc2, fc3)")
  parser.add_argument("--fig_type", default="pdf", type=str, 
                      help="figure file extension type")
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
    print(f'parsing file {ckpt_dir}')
    
    results = []
    for probe_layer in range(0,12):
    
      mask_wq = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/query/mask-o:0')
      mask_wk = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/key/mask-o:0')
      mask_wv = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/value/mask-o:0')
      mask_fc1 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/output/dense/mask-o:0')
      mask_fc2 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/intermediate/dense/mask-o:0')
      mask_fc3 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/output/dense/mask-o:0')    

      wq = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/query/kernel:0')
      wk = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/key/kernel:0')
      wv = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/self/value/kernel:0')

      fc1 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/attention/output/dense/kernel:0')
      fc2 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/intermediate/dense/kernel:0')
      fc3 = sess.graph.get_tensor_by_name(f'bert/encoder/layer_{probe_layer}/output/dense/kernel:0')


      result = sess.run([wq,wk,wv,fc1,fc2,fc3,mask_wq,mask_wk,mask_wv,mask_fc1,mask_fc2,mask_fc3])
      results.append(result)

  font_size=18
  plot_range = 768
  fig_size = 20

  mat_name_to_index = {
    'wq':0,
    'wk':1,
    'wv':2,
    'fc1':3,
    'fc2':4,
    'fc3':5
  }
  layer = args.layer_index
  mat_idx = mat_name_to_index[args.matrix_name]
  print(f'processing layer: {layer} matrix: {args.matrix_name} of ckpt: {args.ckpt_dir}')

  fig = plt.figure(figsize=(fig_size,fig_size))
  ax = fig.add_subplot(1,1,1)

  pruned = np.ma.masked_where(result[mat_idx+6][:plot_range,:plot_range],result[mat_idx][:plot_range,:plot_range])
  remained = np.ma.masked_where(np.logical_not(result[mat_idx+6][:plot_range,:plot_range]),result[mat_idx][:plot_range,:plot_range])

  #cax = ax.matshow(pruned,cmap=plt.get_cmap('Greys'),interpolation='None')
  cax = ax.matshow(remained,cmap=plt.get_cmap('seismic'),interpolation='None')


  plt.xticks(range(0,769,64),fontsize=font_size)
  plt.yticks(range(0,769,64),fontsize=font_size)

  cb = fig.colorbar(cax,fraction=0.035)
  cb.ax.tick_params(labelsize=font_size)
  print(f'saving image to {args.output_dir+"."+args.fig_type}')
  plt.savefig(args.output_dir+"."+args.fig_type,format=args.fig_type)  

  # plt.show()
  plt.close()