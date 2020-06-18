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
    
    prune_layer_wise = False



    if pruning_type == 19 or pruning_type == 20:
      # combine G = 1 and G = granularity MIX pruning_type type == 8
      # combine G = 1 and G = granularity
      # granularity pruning on whole layer

          # prune N + K dimension on whole network
      # N dimension pruning whole column
      # K dimension pruning uses granularity pruning
      # prune N dimesnion first is necessary, and hence consider the pruning on K and N independently. 
      
      # pruning on N
      l2_norm_list = [ np.linalg.norm(score, axis=0) / score.shape[0] for score in accumulated_scores ]


      ###############
      ## To skip the Densor Blocks.
      ###############
      # Top-300 Blocks with G = 64

      skip_block = []
      
      is_right = False
      if skip_block_dim == 64:
        if task_name == 'SQUAD':
          print("\n\n\n SQUAD \n\n\n")
          is_right = True
          skip_block = [1193, 1205, 1098, 1086,  988,  976, 1217, 1096, 1097,  991, 1208,
       1200, 1085, 1196, 1099, 1084, 1112, 1212, 1219, 1291, 1295,  979,
       1108, 1288, 1294, 1110, 1267, 1293, 1215, 1285, 1290, 1289, 1287,
       1292, 1254, 1284,  989, 1286, 1087, 1240, 1252, 1255, 1188, 1218,
       1265, 1280, 1274, 1264, 1268, 1207, 1279, 1281, 1250, 1000, 1241,
         18, 1266,  977, 1272, 1256, 1253, 1273, 1283, 1245, 1220, 1259,
       1275, 1238, 1261, 1282, 1257, 1276, 1093, 1260, 1248, 1278, 1003,
       1239, 1270, 1262, 1109, 1247, 1100, 1244, 1242, 1249, 1263, 1246,
       1258, 1251, 1236, 1235, 1230, 1231, 1271, 1226, 1224, 1227, 1277,
       1203, 1111, 1229,   30, 1225, 1234, 1233, 1232,   16, 1088, 1228,
       1269, 1237, 1243, 1178, 1214, 1186, 1183, 1223, 1182, 1106, 1176,
          6,   28, 1177, 1184, 1185, 1187, 1156, 1136, 1191, 1181, 1179,
       1180,  985, 1133,   19, 1139, 1163, 1195, 1140, 1167, 1160, 1161,
       1146, 1081, 1152, 1138, 1166, 1159, 1145,  994, 1105, 1173, 1132,
       1118, 1129, 1148,  123, 1123, 1158, 1153, 1149, 1150, 1162, 1143,
       1116, 1126, 1122, 1151,   12,  973, 1124, 1155, 1164, 1119, 1117,
       1125, 1114, 1121,  111, 1175, 1157, 1131, 1127, 1168, 1120, 1165,
       1128, 1134, 1130, 1170, 1216, 1171, 1141,    4, 1174, 1154, 1137,
          7, 1147, 1222, 1211, 1070, 1135, 1206,  243, 1169, 1075, 1078,
       1172,  982,  141, 1076, 1058, 1068,   20, 1079, 1072, 1077, 1144,
       1142, 1069, 1074,  997, 1071, 1073, 1045,   13,  134, 1049, 1065,
          0, 1104, 1028,  445,    8,  457, 1057, 1001,  120, 1030, 1027,
       1042, 1044, 1023, 1064, 1036,  340, 1062, 1034,  140,  433, 1061,
       1046, 1032,    1, 1038, 1035, 1047, 1025, 1056, 1199, 1024, 1033,
         31, 1031, 1050, 1043, 1052, 1037, 1054, 1040, 1041, 1026,  328,
       1063, 1053, 1048, 1060, 1039, 1194, 1051, 1029, 1020, 1066, 1055,
       1202, 1021,  884, 1006, 1059,  108, 1213,  132,   24,  962,  967,
        969, 1010, 1022]

        if task_name == 'MNLI':
          print("\n\n\n MNLI \n\n\n")
          is_right = True
          skip_block = [1195, 1207, 1098, 1086, 1097, 1085, 1263, 1278, 1264, 1238, 1294,
        1268, 1250, 1284, 1262, 1276, 1288, 1292, 1295, 1290, 1242, 1274,
        1287, 1291, 1289, 1293, 1272, 1286, 1256, 1285, 1273, 1240, 1258,
        1253, 1244, 1261, 1239, 1281, 1246, 1283, 1270, 1241, 1251, 1279,
        1245, 1243, 1254, 1266, 1271, 1255, 1109, 1267, 1280, 1260, 1248,
        1265, 1257, 1110, 1236, 1237, 1269, 1249, 1282, 1252, 1247, 1259,
          16, 1193,   12, 1277, 1275, 1219,  445,  993,  138,  340,    0,
          433,   18, 1136, 1146, 1205, 1161,  981, 1132,  126,    4,  123,
        1188, 1165,  111, 1140,   19, 1160, 1153, 1138, 1147,  233,   28,
          120, 1150, 1145, 1139, 1131, 1166, 1168, 1130,   30, 1217, 1162,
          114, 1143, 1215, 1186, 1142, 1163, 1184,  328, 1173, 1156, 1167,
        1158, 1133, 1005, 1183, 1155, 1170, 1164, 1182, 1175, 1154, 1148,
        1187,    6, 1172, 1180, 1234, 1129, 1149, 1137, 1185, 1200, 1159,
        1152, 1151, 1178, 1177, 1181, 1179, 1176, 1174, 1203,  108, 1128,
        1144, 1171,  763, 1157, 1224,  243,    7, 1141, 1135, 1228,  221,
          775, 1230, 1232, 1134, 1191,   20, 1229, 1231, 1227, 1235, 1225,
        1114, 1233, 1226,    8,  134, 1105,  141, 1169, 1126, 1127,  127,
        1107, 1124, 1199, 1218, 1122, 1125, 1121, 1118, 1123, 1116,  140,
        1117, 1120, 1108, 1112, 1119,  132, 1211,  115, 1216, 1106,   13,
        1093, 1084,  779, 1223,  767, 1194, 1081,  457, 1212,    1, 1096,
          787,  125,   24,   31, 1060, 1058, 1206,  113, 1076, 1078, 1056,
        1074, 1028, 1039,  232, 1047,  976, 1041, 1025, 1027, 1038, 1075,
        1079, 1077, 1052, 1070, 1034, 1071, 1068,  236, 1072, 1073, 1057,
        1069,  137,  244, 1023,  791, 1043, 1044, 1032, 1065,  352, 1051,
        1062, 1214,  128, 1054,  245, 1021,  554, 1050, 1030,  988, 1029,
        1066,  985, 1036, 1220,  225, 1059, 1049, 1031, 1064, 1104,  758,
          452, 1053,  220,  973,  987, 1055,  135, 1033, 1026, 1042, 1196,
          770, 1040, 1046, 1035, 1037, 1024, 1045, 1048,  247,  224, 1213,
          153, 1022, 1208, 1020,  542,  155,  683,  338,  237, 1111, 1063,
        1067,  242,  440,  882, 1061]

      if skip_block_dim == 32:
        skip_block = [2488, 2173, 2172, 2197, 2390, 2391, 2415, 2414, 2196, 2501, 2537,
        2491, 2552, 2535, 2560, 2477, 2483, 2492, 2485, 2544, 2494, 2588,
        2504, 2568, 2481, 2589, 2534, 2489, 2194, 2548, 2536, 2590, 2576,
        2578, 2573, 2522, 2556, 2545, 2575, 2577, 2582, 2569, 2547, 2512,
        2170, 2591, 2574, 2524, 2549, 2195, 2586, 2585, 2583, 2570, 2565,
        2526, 2580, 2528, 2476, 2581, 2587, 2479, 2571, 2584, 2559, 2554,
        2579, 2509, 2507, 2496, 2484, 2525, 2572, 2527, 2567, 2486, 2523,
        2475, 2498, 2558, 2561, 2553, 2478, 2540, 2171, 2502, 2562, 2221,
        2515, 2516, 2531, 2506, 2533, 2557, 2529, 2541, 2480, 2563, 2546,
        2513, 2503, 2511, 2490, 2521, 2493, 2497, 2566, 2542, 2500, 2514,
        2539, 2510, 2532, 2472, 2482, 2487, 2517, 2220, 2518, 2530, 2520,
        2543, 2508, 2473, 2387, 2499,   25, 2555, 2219, 2474, 2538, 2386,
          33,   32,  277, 2505, 2218, 2322, 2519, 2438, 2290, 2550, 2439,
          24, 2551,   36, 2564,    1,  890,  680, 2343,   37,  466, 2316,
        2264, 2292,  276, 2495, 2411, 2280,  681,  866, 2300, 2376, 2377,
        2410, 2273,  246, 1987, 2306,    0,    9, 2351, 2331, 2272,  252,
          442, 2265, 2281,  891, 2318, 2295, 1986, 2330, 2336,  253,   39,
          222, 2298, 2307, 2293, 2401, 2333, 2375, 2287, 2324,  467, 2320,
        2335, 2327, 2313, 2276, 1963, 2258, 2348, 2340,   57,  867, 2291,
        2309,    8, 2302, 2285,  656, 2323, 2332,   13, 1962, 2334, 2367,
        2279,   12, 2345, 2268, 2369,  240, 2262, 2283,   60, 2372,  247,
        2364, 2312, 2256, 2277, 2267, 2260, 2350, 2311, 2358, 2259,  228,
        2370, 2373, 2468, 2362, 2329, 2352, 2284, 2274, 2346, 2315, 2261,
        2341, 2360, 2319, 2448, 2278, 2321, 2294, 2328, 2354, 2304,   15,
        2368, 2359, 2357, 2371, 2299,  229, 2355, 2289, 2365,  223,   61,
          657, 2431,   56, 2469,   38, 2366, 2275, 2297, 2353, 2356, 2361,
        2326, 2310,  241, 2162, 2325, 2270, 2296, 2186, 2347, 2344, 2374,
        2263, 2363, 2224]


      if skip_block_dim == 16:
        skip_block = [4394, 4393, 5112, 4977, 4346, 4345, 4780, 4783, 4976, 5036, 5122,
        5016, 4831, 4828, 5127, 5059, 4347, 5075, 5048, 5002, 5029, 5052,
        4999, 5063, 5072, 4996, 4344, 5068, 5116, 4395, 5119, 4947, 4985,
        4781, 4987, 4782, 4830, 5115, 4982, 4966, 4829, 4981, 5084, 4969,
        4962, 5089, 5105, 5070, 5003, 5102, 4392, 4340, 5009, 4442, 4960,
        4388, 5097, 5098, 5014, 5121, 5131, 4954, 5057, 4965, 5106, 5071,
        5047, 5104, 5074, 4952, 4971, 5039, 4995, 4973, 5120, 4988, 4978,
        4983, 4955, 5041, 5176, 4957, 4975, 5083, 4958, 5136, 5065, 4970,
        5135, 5108, 5180, 4992, 5179, 5162, 5030, 5156, 5170, 5141, 5147,
        4951, 4944, 5005, 5178, 5177, 5165, 5080, 5153, 5138, 4989, 5095,
        5044, 4967, 5012, 5137, 5154, 5024, 5051, 5054, 5150, 5174, 5173,
        4984, 5091, 5159, 5149, 4390, 5181, 5167, 5161, 5182, 5090, 5152,
        5183, 5045, 5169, 5088, 5019, 5021, 5151, 5148, 5077, 5157, 5032,
        4979, 5146, 5155, 5166, 5008, 4391, 5142, 5160, 5143, 5025, 5094,
        5139, 5172, 5145, 5164, 5078, 5023, 5168, 5018, 4963, 5175, 5093,
        5171, 5140, 5158, 5061, 4389, 4342, 5096, 5125, 5144, 5163, 5069,
        4959, 5073, 5109, 4993, 5026, 5050, 5066, 5056, 5130, 4953, 5099,
        5055, 4341, 4343, 5132, 5124, 5011, 5113, 5049, 5015, 5053, 5067,
        5035, 5007, 5134, 5086, 5042, 4950, 5118, 4968, 5000, 4972, 5081,
        4775, 5004, 5033, 5100, 5046, 4956, 5107, 5031, 5006, 5043, 4997,
        5013, 5117, 5027, 4823, 5123, 4440, 5133, 5110, 4443, 5001, 5092,
          555, 4441, 4949, 5062, 5022, 5114, 5082,   66,  553,   73, 5058,
        4961, 5126, 4438, 4994, 5034,   50, 4980,  505, 4986, 4876,    0,
        4645,   48, 5079, 5085, 5087, 5020, 4773, 4436, 4580,   65, 4772,
        5060,   18, 4945, 4878,   51, 5064,   64, 5028, 4753,   25, 4974,
          457, 4964, 4439, 4990, 4636, 5129, 4530,  932, 1780, 4673, 5040,
        4437, 5037, 1361, 4948,    2, 4774,   67, 4584, 5111, 1782, 4529,
        1362,   75, 4686]

      if skip_block_dim == 8:
        if task_name ==  'SQUAD':
          skip_block = [ 8676,  8684,  8683,  9923,  9920, 10233,  8772, 10141,  9974,
        7837,  9918,  9551,  9550,  9549,  8798,  9641,  8899,  9548,
        8791,  7909,  9647,  8690,  8693,  9642, 10075,  8694,  8790,
        9643,  9644,  8789,  9984,  8692,  9645,  8788,  8691,  8787,
       10213,  8786,  9971,  9646,  7933,  9547,  9910,  8681,  8774,
       10143,  8702, 10025,  8776,  9506, 10033, 10245,  7815,  7813,
        8703,  8678, 10266,  9759,  7809,  8674,  9754,  8777,  9640,
        9546,  9544,  8679,  8779,  8780, 10048,  9545,  9743,  9742,
        8680, 10005,  8799,  9904, 10081,  9906,  8770,  8695,  7808,
        9935, 10202,  8700,  8785,  8775, 10116,  8689,  8688,  7911,
        9739, 10252,  8687,  9900, 10049,  7917, 10198,  8784, 10224,
       10035,  7904, 10008,  7905,  9602,  8796,  8773,  7907,  9753,
        7814, 10152,  8783, 10110, 10253, 10129,  9722,  9724,  9998,
        9741,  8769, 10007,  8898, 10099,  9938, 10218, 10034,  7910,
        7932, 10006, 10221,  7811,  9740, 10055, 10225,  9890,  9927,
       10183, 10021,  7821, 10207,  7834, 10237,  7918,  7930, 10118,
       10039,  9736,  8673, 10242,  7812,  7914, 10038,  9669,  9702,
        9570, 10216,  7906,  9665,  9668,  9671,  7822, 10136,  9572,
        8794,  9698, 10244, 10258,  9606,  8677,  7928,  8900,  9666,
        9575,  9696,  9697,  7908, 10078,  9600,  8698, 10261, 10137,
        7934,  8896,  7931,  7935,  7912,  9908,  7836,  9989,  9745,
        9976,  9921,  9738,  8869,  7810,  9727,  9664,  8781,  9922,
       10016, 10112,  9667,  9603,  9573, 10094,  9755, 10255, 10192,
       10188,  8866,  9569, 10047, 10251,  8868,  7818,  9903, 10115,
        8768,  9751, 10150,  8782,  9670,  9737, 10268, 10120,   147,
       10201, 10132,  9607, 10193,  9601, 10347,  9744, 10172, 10066,
       10361, 10334,  9660,  8778,  7929,  9752,  8881,  9699, 10080,
       10018, 10362, 10262, 10307,  9604, 10056, 10046,  9975, 10063,
         145, 10320, 10308, 10212,  9748, 10287,  7816, 10344,  8870,
        9978, 10174, 10342, 10040, 10340, 10331,  9917, 10366,  8795,
        8882, 10284,  8771, 10282,  8886, 10358, 10345, 10309, 10332,
       10302,  7835,  8897, 10138, 10097, 10319, 10311, 10304, 10324,
       10329,  9721, 10113, 10169, 10190,  7913,  8885,  9605,  9912,
       10359,  9568, 10305]

        if task_name == 'MNLI':
          skip_block = [ 9970,  9954,  9955,  9562, 10168,  9957,  9561,  9560,  9962,
        10161,  9566, 10151, 10144, 10142,  8790,  8789,  8788,  8787,
          8786, 10135,  9979, 10149,  9567, 10174,  9953, 10245, 10238,
          9924,  9929, 10249,  9908, 10225, 10224, 10216, 10214, 10213,
        10211, 10209,  9943, 10194,  9898,  9894,  9889,  9663, 10181,
          9658,  8780, 10126, 10240, 10123, 10097,  8690, 10002,  8691,
          9999,  8684, 10102, 10082,  8692,  8693, 10073, 10070, 10104,
        10112, 10019, 10032, 10059, 10045, 10055, 10118,  9660,  9918,
          9926, 10004, 10049, 10184, 10262,  9935, 10254, 10255, 10080,
        10271,  9901,  9940, 10092, 10268, 10072, 10005, 10095,  9952,
        10033, 10137, 10109,  9974,  9992,  9657,  8695,  8694, 10244,
          9976,  9966, 10132, 10156,  9996,  9656, 10167,  9564, 10086,
        10231, 10232, 10096, 10247, 10141,  9964,  8777,  8689, 10119,
          9662, 10150, 10120, 10183,  9947, 10025, 10233, 10040, 10129,
          9921, 10176, 10139,  9938, 10015, 10011,  9993, 10189,  9959,
        10098, 10125,  9932, 10105, 10058,  9902,  8681,  8688,  9915,
          9991, 10204, 10051,  8885, 10145, 10006, 10077, 10136, 10114,
        10127,  9998, 10265,  9904,  8791, 10266, 10239, 10057, 10227,
        10179,  9895,  9975,  9565, 10085,  8784,  9965, 10152,  9971,
        10065,  9563, 10178,  9931,  9933,  9906, 10230, 10029, 10172,
        10039,  9661, 10242,  8785,  9939, 10079, 10199, 10197,  9945,
        10221, 10022, 10007, 10052, 10163,  9916,  9659,  9936, 10205,
        10196, 10191,  8884, 10140,  9963,  9986, 10169, 10146,  8680,
          8687,  9920,  9925, 10192,  9985,  8683,  9951, 10028, 10116,
        10257, 10210,  9911, 10031,  9969, 10200, 10243, 10130, 10234,
        10016,  8776, 10294, 10066, 10018, 10353,  9980, 10261, 10283,
        10360, 10341, 10042, 10106, 10195, 10272, 10062, 10356, 10352,
        10276,  9930,  9910, 10328, 10324, 10359, 10155, 10078, 10026,
        10300,  9949, 10115, 10090, 10000, 10061, 10273,  9905, 10312,
        10299,  9950, 10237, 10318, 10263, 10355, 10331, 10307,  9909,
        10308, 10315, 10347, 10111,  8779,  9990, 10131, 10274, 10088,
        10094, 10345, 10060, 10313, 10304, 10047, 10219,  8783, 10148,
        10212, 10208,  9942, 10143, 10364, 10337, 10358, 10354, 10349,
        10330, 10325, 10089]

      lyaer_dim = (768 * (4 + 4 + 1)) / skip_block_dim
      block_dim = 768 / skip_block_dim
      block_num = 0

      for block_id in skip_block:
        if block_remain == 0:
          return
        layer_out = (int)(block_id / lyaer_dim)
        layer_in = (int)(block_id % lyaer_dim)
        block = 0
        layer = 0
        if layer_in < block_dim * 4:
          ## 0 - 3 layers
          layer = (int) (layer_in / block_dim)
          block = (int) (layer_in % block_dim)
        elif layer_in < block_dim * 8:
          ## 4 layer
          layer = 4
          block = layer_in - block_dim * 4
        elif layer_in < lyaer_dim:
          ## 5 layer
          layer = 5
          block = layer_in - block_dim * 8
        
        layer = layer_out * 6 + layer

        layer_list = l2_norm_list[layer]
        start = int(block * skip_block_dim)
        end = int((block+1) * skip_block_dim)
        layer_list[start:end] = 0

        block_num += 1
        if block_num >= block_remain:
          break


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

      ## g1percent
      threshold = np.percentile(split_vector_l2_norm_list, sparsity + g1percent)

      layer_mask = []

      for layer in range(len(pruned_accumulated_scores)):
        score = pruned_accumulated_scores[layer]
        split_score = split_score_list[layer]
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ mask.reshape(mask.shape[0], 1) for mask in split_mask ]
        split_mask = [ np.tile(m, (1, split_score[i].shape[1])).reshape(split_score[i].shape) for i, m in enumerate(split_mask) ]
        mask = np.concatenate(split_mask, axis = 1)
        
        assert(mask.shape == tiled_mask[layer].shape)
        assert(tiled_mask[layer].shape == placeholders[layer].shape)
        #feed[placeholders[layer]] = mask * tiled_mask[layer]
        layer_mask.append(mask * tiled_mask[layer])

      if pruning_type == 20:
        with open('accumulated_scores_0.pkl', 'rb') as f:
            accumulated_scores = pickle.load(f)


      # For G = 1
      l2_norm_list = []
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        mask = layer_mask[layer]
        split_l2_norm = split_l2_norm - (mask.transpose())
        l2_norm_list.append(split_l2_norm)
      
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]


      threshold = np.percentile(split_weight_l2_norm_list, 100 - g1percent)

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(1)], axis=1) for m in split_mask ]
        mask1 = np.concatenate(split_mask, axis = 1)
        mask0 = layer_mask[layer]
        mask = mask0 + mask1
        assert(mask.shape == placeholders[layer].shape)
        feed[placeholders[layer]] = mask

        
      sess.run(assign_ops, feed)
      return


    if pruning_type == 18:
      # combine G = 1 and G = granularity MIX pruning_type type == 7
      # granularity pruning on whole layer
      l2_norm_list = []
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1] / granularity
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        l2_norm_list.append(split_l2_norm)
      
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]

      
      threshold = np.percentile(split_weight_l2_norm_list, sparsity + g1percent)

      layer_mask = []

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1] / granularity
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(granularity)], axis=1) for m in split_mask ]
        mask = np.concatenate(split_mask, axis = 1)
        assert(mask.shape == placeholders[layer].shape)
        layer_mask.append(mask)
        #feed[placeholders[layer]] = mask

      # For G = 1
      l2_norm_list = []
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        mask = layer_mask[layer]
        split_l2_norm = split_l2_norm - (mask.transpose())
        l2_norm_list.append(split_l2_norm)
      
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]


      threshold = np.percentile(split_weight_l2_norm_list, 100 - g1percent)

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(1)], axis=1) for m in split_mask ]
        mask1 = np.concatenate(split_mask, axis = 1)
        mask0 = layer_mask[layer]
        mask = mask0 + mask1
        assert(mask.shape == placeholders[layer].shape)
        feed[placeholders[layer]] = mask
      sess.run(assign_ops, feed)
      return



    if pruning_type == 17:
      # combine G = 1 and G = granularity
      # granularity pruning on whole layer
      l2_norm_list = []
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1] / granularity
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        l2_norm_list.append(split_l2_norm)
      
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]

      g2percent = 100 - g1percent
      threshold = np.percentile(split_weight_l2_norm_list, sparsity * (g2percent / 100))

      layer_mask = []

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1] / granularity
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(granularity)], axis=1) for m in split_mask ]
        mask = np.concatenate(split_mask, axis = 1)
        assert(mask.shape == placeholders[layer].shape)
        layer_mask.append(mask)
        #feed[placeholders[layer]] = mask

      # For G = 1
      l2_norm_list = []
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        mask = layer_mask[layer]
        split_l2_norm = split_l2_norm - (~ mask.transpose())
        l2_norm_list.append(split_l2_norm)
      
      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]


      threshold = np.percentile(split_weight_l2_norm_list, sparsity)

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(1)], axis=1) for m in split_mask ]
        mask = np.concatenate(split_mask, axis = 1)
        assert(mask.shape == placeholders[layer].shape)
        feed[placeholders[layer]] = mask
      sess.run(assign_ops, feed)
      return


    if pruning_type == 16:
          # prune N + K dimension on whole network
      # N dimension pruning whole column
      # K dimension pruning uses granularity pruning
      # prune N dimesnion first is necessary, and hence consider the pruning on K and N independently. 
      
      # pruning on N
      l2_norm_list = [ np.linalg.norm(score, axis=0) / score.shape[0] for score in accumulated_scores ]


      ###############
      ## To skip the Densor Blocks.
      ###############
      # Top-300 Blocks with G = 64

      skip_block = []

      if skip_block_dim == 8:
        skip_block = [10015,  9898,  9901, 10216, 10214, 10213, 10211, 10210, 10209,
        9902, 10205, 10204,  9904, 10196,  9979, 10194,  9906, 10191,
        9962, 10184, 10183,  9908, 10181, 10179, 10174,  9959,  9957,
       10172, 10224,  9894, 10265, 10262, 10261, 10254,  9560,  9561,
        9562, 10253,  9564, 10249,  9566, 10247, 10245, 10244,  9567,
       10022,  9657,  9658, 10240, 10238,  9660,  9662,  9663, 10234,
        9889, 10225, 10268, 10168,  9924, 10090,  9943,  9945, 10082,
       10078, 10073, 10070, 10065,  9951, 10062, 10059, 10058, 10057,
       10055,  9952, 10049,  9953, 10045,  9954, 10040, 10039,  9955,
       10033, 10032, 10028, 10092, 10161, 10095, 10098, 10151, 10150,
       10149,  9925,  9927, 10146,  9929, 10144, 10142, 10139, 10135,
        9933, 10132,  9935, 10127, 10126, 10125, 10123, 10118,  9938,
       10112,  9939,  9940, 10104, 10102, 10097, 10270, 10221,  8789,
        8693, 10011,  9966,  8790,  8684,  8694,  8786, 10271,  8788,
        8787,  9974,  9992,  8776,  9993,  8777, 10019,  8681,  8885,
        8780, 10006,  9998,  8690,  8692,  8680, 10005, 10016,  9970,
        9976,  8691,  9999,  9985, 10002, 10167,  9911,  8779, 10080,
       10156, 10004, 10086,  8695, 10111,  8687, 10096,  9905, 10109,
        9996, 10189, 10170,  9914, 10114, 10251, 10137, 10036,  9963,
       10051,  9926,  9565, 10052, 10148, 10116, 10255,  8683,  9656,
       10068, 10239, 10155, 10266, 10072,  9947,  9918,  9916, 10031,
        8689, 10136,  9932, 10141,  9912, 10143, 10120, 10176, 10163,
       10119, 10115,  7853,  9921, 10178, 10105,  9967, 10190,  9975,
       10233, 10232, 10231,  9563,  9964,  9551,  9949, 10219, 10187,
       10025, 10037,  9942, 10029, 10212, 10152, 10061, 10243, 10129,
        9986, 10263, 10145,  9931,  7945,     4, 10018,  7950, 10130,
       10007,  8688, 10017, 10077,  9915,  9991,  7851,  9971, 10169,
       10066, 10000, 10197, 10140,  7949,  9936,  9659, 10222,  8791,
        9647,  9990,  8783, 10091, 10079, 10048, 10101,  7854,  7849,
       10085,  9545, 10227, 10230,  9895, 10242, 10177,  9661, 10199,
        8784,   890,  9965, 10257,  1110, 10094,   132, 10047,  9950,
        8785,  7946, 10001,  8682, 10106,    36, 10160,  9930,  8884,
        9920, 10100, 10088, 10294, 10200, 10352,  8685,   100, 10272,
       10276,  9909, 10060, 10186,  9549, 10237,  9550, 10360, 10353,
       10283, 10356, 10341, 10359,  7947, 10026,  9888, 10192, 10273,
        9969,  2724,  9645, 10107,   986, 10042, 10089,  1106, 10328,
       10050, 10274, 10318,  9980, 10324,  8782, 10008, 10013,  8781,
       10300, 10299, 10103,     1,  7948,  1109, 10312, 10259, 10354,
       10337, 10323, 10355, 10275, 10347, 10330, 10308, 10195,  1864,
        9956, 10313, 10304, 10358, 10345, 10331, 10307,  9977, 10311,
       10277, 10334, 10325, 10315, 10363, 10332,  9910,  9642, 10364,
       10348, 10361, 10282, 10303, 10349, 10309, 10340, 10188,  9984,
       10366, 10357, 10346, 10297,  8686, 10362, 10290, 10293, 10295,
       10322, 10305, 10317, 10279, 10367, 10321, 10064, 10298, 10335,
       10285, 10320, 10306, 10338]


      lyaer_dim = (768 * (4 + 4 + 1)) / skip_block_dim
      block_dim = 768 / skip_block_dim
      block_num = 0

      for block_id in skip_block:
        if block_remain == 0:
          return
        layer_out = (int)(block_id / lyaer_dim)
        layer_in = (int)(block_id % lyaer_dim)
        block = 0
        layer = 0
        if layer_in < block_dim * 4:
          ## 0 - 3 layers
          layer = (int) (layer_in / block_dim)
          block = (int) (layer_in % block_dim)
        elif layer_in < block_dim * 8:
          ## 4 layer
          layer = 4
          block = layer_in - block_dim * 4
        elif layer_in < lyaer_dim:
          ## 5 layer
          layer = 5
          block = layer_in - block_dim * 8
        
        layer = layer_out * 6 + layer

        layer_list = l2_norm_list[layer]
        start = int(block * skip_block_dim)
        end = int((block+1) * skip_block_dim)
        layer_list[start:end] = 0

        block_num += 1
        if block_num >= block_remain:
          break


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


    if pruning_type == 15:
          # prune N + K dimension on whole network
      # N dimension pruning whole column
      # K dimension pruning uses granularity pruning
      # prune N dimesnion first is necessary, and hence consider the pruning on K and N independently. 
      
      # pruning on N
      l2_norm_list = [ np.linalg.norm(score, axis=0) / score.shape[0] for score in accumulated_scores ]


      ###############
      ## To skip the Densor Blocks.
      ###############
      # Top-300 Blocks with G = 64

      skip_block = []

      if skip_block_dim == 64:
        skip_block = [1193, 1205, 1098, 1086,  988,  976, 1217, 1096, 1097,  991, 1208,
       1200, 1085, 1196, 1099, 1084, 1112, 1212, 1219, 1291, 1295,  979,
       1108, 1288, 1294, 1110, 1267, 1293, 1215, 1285, 1290, 1289, 1287,
       1292, 1254, 1284,  989, 1286, 1087, 1240, 1252, 1255, 1188, 1218,
       1265, 1280, 1274, 1264, 1268, 1207, 1279, 1281, 1250, 1000, 1241,
         18, 1266,  977, 1272, 1256, 1253, 1273, 1283, 1245, 1220, 1259,
       1275, 1238, 1261, 1282, 1257, 1276, 1093, 1260, 1248, 1278, 1003,
       1239, 1270, 1262, 1109, 1247, 1100, 1244, 1242, 1249, 1263, 1246,
       1258, 1251, 1236, 1235, 1230, 1231, 1271, 1226, 1224, 1227, 1277,
       1203, 1111, 1229,   30, 1225, 1234, 1233, 1232,   16, 1088, 1228,
       1269, 1237, 1243, 1178, 1214, 1186, 1183, 1223, 1182, 1106, 1176,
          6,   28, 1177, 1184, 1185, 1187, 1156, 1136, 1191, 1181, 1179,
       1180,  985, 1133,   19, 1139, 1163, 1195, 1140, 1167, 1160, 1161,
       1146, 1081, 1152, 1138, 1166, 1159, 1145,  994, 1105, 1173, 1132,
       1118, 1129, 1148,  123, 1123, 1158, 1153, 1149, 1150, 1162, 1143,
       1116, 1126, 1122, 1151,   12,  973, 1124, 1155, 1164, 1119, 1117,
       1125, 1114, 1121,  111, 1175, 1157, 1131, 1127, 1168, 1120, 1165,
       1128, 1134, 1130, 1170, 1216, 1171, 1141,    4, 1174, 1154, 1137,
          7, 1147, 1222, 1211, 1070, 1135, 1206,  243, 1169, 1075, 1078,
       1172,  982,  141, 1076, 1058, 1068,   20, 1079, 1072, 1077, 1144,
       1142, 1069, 1074,  997, 1071, 1073, 1045,   13,  134, 1049, 1065,
          0, 1104, 1028,  445,    8,  457, 1057, 1001,  120, 1030, 1027,
       1042, 1044, 1023, 1064, 1036,  340, 1062, 1034,  140,  433, 1061,
       1046, 1032,    1, 1038, 1035, 1047, 1025, 1056, 1199, 1024, 1033,
         31, 1031, 1050, 1043, 1052, 1037, 1054, 1040, 1041, 1026,  328,
       1063, 1053, 1048, 1060, 1039, 1194, 1051, 1029, 1020, 1066, 1055,
       1202, 1021,  884, 1006, 1059,  108, 1213,  132,   24,  962,  967,
        969, 1010, 1022]
        # [1195, 1207, 1098, 1086, 1097, 1085, 1263, 1278, 1264, 1238, 1294,
        # 1268, 1250, 1284, 1262, 1276, 1288, 1292, 1295, 1290, 1242, 1274,
        # 1287, 1291, 1289, 1293, 1272, 1286, 1256, 1285, 1273, 1240, 1258,
        # 1253, 1244, 1261, 1239, 1281, 1246, 1283, 1270, 1241, 1251, 1279,
        # 1245, 1243, 1254, 1266, 1271, 1255, 1109, 1267, 1280, 1260, 1248,
        # 1265, 1257, 1110, 1236, 1237, 1269, 1249, 1282, 1252, 1247, 1259,
        #   16, 1193,   12, 1277, 1275, 1219,  445,  993,  138,  340,    0,
        #   433,   18, 1136, 1146, 1205, 1161,  981, 1132,  126,    4,  123,
        # 1188, 1165,  111, 1140,   19, 1160, 1153, 1138, 1147,  233,   28,
        #   120, 1150, 1145, 1139, 1131, 1166, 1168, 1130,   30, 1217, 1162,
        #   114, 1143, 1215, 1186, 1142, 1163, 1184,  328, 1173, 1156, 1167,
        # 1158, 1133, 1005, 1183, 1155, 1170, 1164, 1182, 1175, 1154, 1148,
        # 1187,    6, 1172, 1180, 1234, 1129, 1149, 1137, 1185, 1200, 1159,
        # 1152, 1151, 1178, 1177, 1181, 1179, 1176, 1174, 1203,  108, 1128,
        # 1144, 1171,  763, 1157, 1224,  243,    7, 1141, 1135, 1228,  221,
        #   775, 1230, 1232, 1134, 1191,   20, 1229, 1231, 1227, 1235, 1225,
        # 1114, 1233, 1226,    8,  134, 1105,  141, 1169, 1126, 1127,  127,
        # 1107, 1124, 1199, 1218, 1122, 1125, 1121, 1118, 1123, 1116,  140,
        # 1117, 1120, 1108, 1112, 1119,  132, 1211,  115, 1216, 1106,   13,
        # 1093, 1084,  779, 1223,  767, 1194, 1081,  457, 1212,    1, 1096,
        #   787,  125,   24,   31, 1060, 1058, 1206,  113, 1076, 1078, 1056,
        # 1074, 1028, 1039,  232, 1047,  976, 1041, 1025, 1027, 1038, 1075,
        # 1079, 1077, 1052, 1070, 1034, 1071, 1068,  236, 1072, 1073, 1057,
        # 1069,  137,  244, 1023,  791, 1043, 1044, 1032, 1065,  352, 1051,
        # 1062, 1214,  128, 1054,  245, 1021,  554, 1050, 1030,  988, 1029,
        # 1066,  985, 1036, 1220,  225, 1059, 1049, 1031, 1064, 1104,  758,
        #   452, 1053,  220,  973,  987, 1055,  135, 1033, 1026, 1042, 1196,
        #   770, 1040, 1046, 1035, 1037, 1024, 1045, 1048,  247,  224, 1213,
        #   153, 1022, 1208, 1020,  542,  155,  683,  338,  237, 1111, 1063,
        # 1067,  242,  440,  882, 1061]
      
      if skip_block_dim == 32:
        skip_block = [2488, 2173, 2172, 2197, 2390, 2391, 2415, 2414, 2196, 2501, 2537,
        2491, 2552, 2535, 2560, 2477, 2483, 2492, 2485, 2544, 2494, 2588,
        2504, 2568, 2481, 2589, 2534, 2489, 2194, 2548, 2536, 2590, 2576,
        2578, 2573, 2522, 2556, 2545, 2575, 2577, 2582, 2569, 2547, 2512,
        2170, 2591, 2574, 2524, 2549, 2195, 2586, 2585, 2583, 2570, 2565,
        2526, 2580, 2528, 2476, 2581, 2587, 2479, 2571, 2584, 2559, 2554,
        2579, 2509, 2507, 2496, 2484, 2525, 2572, 2527, 2567, 2486, 2523,
        2475, 2498, 2558, 2561, 2553, 2478, 2540, 2171, 2502, 2562, 2221,
        2515, 2516, 2531, 2506, 2533, 2557, 2529, 2541, 2480, 2563, 2546,
        2513, 2503, 2511, 2490, 2521, 2493, 2497, 2566, 2542, 2500, 2514,
        2539, 2510, 2532, 2472, 2482, 2487, 2517, 2220, 2518, 2530, 2520,
        2543, 2508, 2473, 2387, 2499,   25, 2555, 2219, 2474, 2538, 2386,
          33,   32,  277, 2505, 2218, 2322, 2519, 2438, 2290, 2550, 2439,
          24, 2551,   36, 2564,    1,  890,  680, 2343,   37,  466, 2316,
        2264, 2292,  276, 2495, 2411, 2280,  681,  866, 2300, 2376, 2377,
        2410, 2273,  246, 1987, 2306,    0,    9, 2351, 2331, 2272,  252,
          442, 2265, 2281,  891, 2318, 2295, 1986, 2330, 2336,  253,   39,
          222, 2298, 2307, 2293, 2401, 2333, 2375, 2287, 2324,  467, 2320,
        2335, 2327, 2313, 2276, 1963, 2258, 2348, 2340,   57,  867, 2291,
        2309,    8, 2302, 2285,  656, 2323, 2332,   13, 1962, 2334, 2367,
        2279,   12, 2345, 2268, 2369,  240, 2262, 2283,   60, 2372,  247,
        2364, 2312, 2256, 2277, 2267, 2260, 2350, 2311, 2358, 2259,  228,
        2370, 2373, 2468, 2362, 2329, 2352, 2284, 2274, 2346, 2315, 2261,
        2341, 2360, 2319, 2448, 2278, 2321, 2294, 2328, 2354, 2304,   15,
        2368, 2359, 2357, 2371, 2299,  229, 2355, 2289, 2365,  223,   61,
          657, 2431,   56, 2469,   38, 2366, 2275, 2297, 2353, 2356, 2361,
        2326, 2310,  241, 2162, 2325, 2270, 2296, 2186, 2347, 2344, 2374,
        2263, 2363, 2224]


      if skip_block_dim == 16:
        skip_block = [4394, 4393, 5112, 4977, 4346, 4345, 4780, 4783, 4976, 5036, 5122,
        5016, 4831, 4828, 5127, 5059, 4347, 5075, 5048, 5002, 5029, 5052,
        4999, 5063, 5072, 4996, 4344, 5068, 5116, 4395, 5119, 4947, 4985,
        4781, 4987, 4782, 4830, 5115, 4982, 4966, 4829, 4981, 5084, 4969,
        4962, 5089, 5105, 5070, 5003, 5102, 4392, 4340, 5009, 4442, 4960,
        4388, 5097, 5098, 5014, 5121, 5131, 4954, 5057, 4965, 5106, 5071,
        5047, 5104, 5074, 4952, 4971, 5039, 4995, 4973, 5120, 4988, 4978,
        4983, 4955, 5041, 5176, 4957, 4975, 5083, 4958, 5136, 5065, 4970,
        5135, 5108, 5180, 4992, 5179, 5162, 5030, 5156, 5170, 5141, 5147,
        4951, 4944, 5005, 5178, 5177, 5165, 5080, 5153, 5138, 4989, 5095,
        5044, 4967, 5012, 5137, 5154, 5024, 5051, 5054, 5150, 5174, 5173,
        4984, 5091, 5159, 5149, 4390, 5181, 5167, 5161, 5182, 5090, 5152,
        5183, 5045, 5169, 5088, 5019, 5021, 5151, 5148, 5077, 5157, 5032,
        4979, 5146, 5155, 5166, 5008, 4391, 5142, 5160, 5143, 5025, 5094,
        5139, 5172, 5145, 5164, 5078, 5023, 5168, 5018, 4963, 5175, 5093,
        5171, 5140, 5158, 5061, 4389, 4342, 5096, 5125, 5144, 5163, 5069,
        4959, 5073, 5109, 4993, 5026, 5050, 5066, 5056, 5130, 4953, 5099,
        5055, 4341, 4343, 5132, 5124, 5011, 5113, 5049, 5015, 5053, 5067,
        5035, 5007, 5134, 5086, 5042, 4950, 5118, 4968, 5000, 4972, 5081,
        4775, 5004, 5033, 5100, 5046, 4956, 5107, 5031, 5006, 5043, 4997,
        5013, 5117, 5027, 4823, 5123, 4440, 5133, 5110, 4443, 5001, 5092,
          555, 4441, 4949, 5062, 5022, 5114, 5082,   66,  553,   73, 5058,
        4961, 5126, 4438, 4994, 5034,   50, 4980,  505, 4986, 4876,    0,
        4645,   48, 5079, 5085, 5087, 5020, 4773, 4436, 4580,   65, 4772,
        5060,   18, 4945, 4878,   51, 5064,   64, 5028, 4753,   25, 4974,
          457, 4964, 4439, 4990, 4636, 5129, 4530,  932, 1780, 4673, 5040,
        4437, 5037, 1361, 4948,    2, 4774,   67, 4584, 5111, 1782, 4529,
        1362,   75, 4686]

      if skip_block_dim == 8:
        skip_block = [ 8676,  8684,  8683,  9923,  9920, 10233,  8772, 10141,  9974,
        7837,  9918,  9551,  9550,  9549,  8798,  9641,  8899,  9548,
        8791,  7909,  9647,  8690,  8693,  9642, 10075,  8694,  8790,
        9643,  9644,  8789,  9984,  8692,  9645,  8788,  8691,  8787,
       10213,  8786,  9971,  9646,  7933,  9547,  9910,  8681,  8774,
       10143,  8702, 10025,  8776,  9506, 10033, 10245,  7815,  7813,
        8703,  8678, 10266,  9759,  7809,  8674,  9754,  8777,  9640,
        9546,  9544,  8679,  8779,  8780, 10048,  9545,  9743,  9742,
        8680, 10005,  8799,  9904, 10081,  9906,  8770,  8695,  7808,
        9935, 10202,  8700,  8785,  8775, 10116,  8689,  8688,  7911,
        9739, 10252,  8687,  9900, 10049,  7917, 10198,  8784, 10224,
       10035,  7904, 10008,  7905,  9602,  8796,  8773,  7907,  9753,
        7814, 10152,  8783, 10110, 10253, 10129,  9722,  9724,  9998,
        9741,  8769, 10007,  8898, 10099,  9938, 10218, 10034,  7910,
        7932, 10006, 10221,  7811,  9740, 10055, 10225,  9890,  9927,
       10183, 10021,  7821, 10207,  7834, 10237,  7918,  7930, 10118,
       10039,  9736,  8673, 10242,  7812,  7914, 10038,  9669,  9702,
        9570, 10216,  7906,  9665,  9668,  9671,  7822, 10136,  9572,
        8794,  9698, 10244, 10258,  9606,  8677,  7928,  8900,  9666,
        9575,  9696,  9697,  7908, 10078,  9600,  8698, 10261, 10137,
        7934,  8896,  7931,  7935,  7912,  9908,  7836,  9989,  9745,
        9976,  9921,  9738,  8869,  7810,  9727,  9664,  8781,  9922,
       10016, 10112,  9667,  9603,  9573, 10094,  9755, 10255, 10192,
       10188,  8866,  9569, 10047, 10251,  8868,  7818,  9903, 10115,
        8768,  9751, 10150,  8782,  9670,  9737, 10268, 10120,   147,
       10201, 10132,  9607, 10193,  9601, 10347,  9744, 10172, 10066,
       10361, 10334,  9660,  8778,  7929,  9752,  8881,  9699, 10080,
       10018, 10362, 10262, 10307,  9604, 10056, 10046,  9975, 10063,
         145, 10320, 10308, 10212,  9748, 10287,  7816, 10344,  8870,
        9978, 10174, 10342, 10040, 10340, 10331,  9917, 10366,  8795,
        8882, 10284,  8771, 10282,  8886, 10358, 10345, 10309, 10332,
       10302,  7835,  8897, 10138, 10097, 10319, 10311, 10304, 10324,
       10329,  9721, 10113, 10169, 10190,  7913,  8885,  9605,  9912,
       10359,  9568, 10305]
      #  [ 9970,  9954,  9955,  9562, 10168,  9957,  9561,  9560,  9962,
      #   10161,  9566, 10151, 10144, 10142,  8790,  8789,  8788,  8787,
      #     8786, 10135,  9979, 10149,  9567, 10174,  9953, 10245, 10238,
      #     9924,  9929, 10249,  9908, 10225, 10224, 10216, 10214, 10213,
      #   10211, 10209,  9943, 10194,  9898,  9894,  9889,  9663, 10181,
      #     9658,  8780, 10126, 10240, 10123, 10097,  8690, 10002,  8691,
      #     9999,  8684, 10102, 10082,  8692,  8693, 10073, 10070, 10104,
      #   10112, 10019, 10032, 10059, 10045, 10055, 10118,  9660,  9918,
      #     9926, 10004, 10049, 10184, 10262,  9935, 10254, 10255, 10080,
      #   10271,  9901,  9940, 10092, 10268, 10072, 10005, 10095,  9952,
      #   10033, 10137, 10109,  9974,  9992,  9657,  8695,  8694, 10244,
      #     9976,  9966, 10132, 10156,  9996,  9656, 10167,  9564, 10086,
      #   10231, 10232, 10096, 10247, 10141,  9964,  8777,  8689, 10119,
      #     9662, 10150, 10120, 10183,  9947, 10025, 10233, 10040, 10129,
      #     9921, 10176, 10139,  9938, 10015, 10011,  9993, 10189,  9959,
      #   10098, 10125,  9932, 10105, 10058,  9902,  8681,  8688,  9915,
      #     9991, 10204, 10051,  8885, 10145, 10006, 10077, 10136, 10114,
      #   10127,  9998, 10265,  9904,  8791, 10266, 10239, 10057, 10227,
      #   10179,  9895,  9975,  9565, 10085,  8784,  9965, 10152,  9971,
      #   10065,  9563, 10178,  9931,  9933,  9906, 10230, 10029, 10172,
      #   10039,  9661, 10242,  8785,  9939, 10079, 10199, 10197,  9945,
      #   10221, 10022, 10007, 10052, 10163,  9916,  9659,  9936, 10205,
      #   10196, 10191,  8884, 10140,  9963,  9986, 10169, 10146,  8680,
      #     8687,  9920,  9925, 10192,  9985,  8683,  9951, 10028, 10116,
      #   10257, 10210,  9911, 10031,  9969, 10200, 10243, 10130, 10234,
      #   10016,  8776, 10294, 10066, 10018, 10353,  9980, 10261, 10283,
      #   10360, 10341, 10042, 10106, 10195, 10272, 10062, 10356, 10352,
      #   10276,  9930,  9910, 10328, 10324, 10359, 10155, 10078, 10026,
      #   10300,  9949, 10115, 10090, 10000, 10061, 10273,  9905, 10312,
      #   10299,  9950, 10237, 10318, 10263, 10355, 10331, 10307,  9909,
      #   10308, 10315, 10347, 10111,  8779,  9990, 10131, 10274, 10088,
      #   10094, 10345, 10060, 10313, 10304, 10047, 10219,  8783, 10148,
      #   10212, 10208,  9942, 10143, 10364, 10337, 10358, 10354, 10349,
      #   10330, 10325, 10089]

      lyaer_dim = (768 * (4 + 4 + 1)) / skip_block_dim
      block_dim = 768 / skip_block_dim
      block_num = 0

      for block_id in skip_block:
        if block_remain == 0:
          return
        layer_out = (int)(block_id / lyaer_dim)
        layer_in = (int)(block_id % lyaer_dim)
        block = 0
        layer = 0
        if layer_in < block_dim * 4:
          ## 0 - 3 layers
          layer = (int) (layer_in / block_dim)
          block = (int) (layer_in % block_dim)
        elif layer_in < block_dim * 8:
          ## 4 layer
          layer = 4
          block = layer_in - block_dim * 4
        elif layer_in < lyaer_dim:
          ## 5 layer
          layer = 5
          block = layer_in - block_dim * 8
        
        layer = layer_out * 6 + layer

        layer_list = l2_norm_list[layer]
        start = int(block * skip_block_dim)
        end = int((block+1) * skip_block_dim)
        layer_list[start:end] = 0

        block_num += 1
        if block_num >= block_remain:
          break


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


    if pruning_type == 14:
      # granularity pruning on whole layer
      # AW = 0 and AW = granularity
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
      

      ###############
      ## To skip the Densor Blocks.
      ###############

      # Top-300 Blocks with G = 64
      skip_block = [1195, 1207, 1098, 1086, 1097, 1085, 1263, 1278, 1264, 1238, 1294,
       1268, 1250, 1284, 1262, 1276, 1288, 1292, 1295, 1290, 1242, 1274,
       1287, 1291, 1289, 1293, 1272, 1286, 1256, 1285, 1273, 1240, 1258,
       1253, 1244, 1261, 1239, 1281, 1246, 1283, 1270, 1241, 1251, 1279,
       1245, 1243, 1254, 1266, 1271, 1255, 1109, 1267, 1280, 1260, 1248,
       1265, 1257, 1110, 1236, 1237, 1269, 1249, 1282, 1252, 1247, 1259,
         16, 1193,   12, 1277, 1275, 1219,  445,  993,  138,  340,    0,
        433,   18, 1136, 1146, 1205, 1161,  981, 1132,  126,    4,  123,
       1188, 1165,  111, 1140,   19, 1160, 1153, 1138, 1147,  233,   28,
        120, 1150, 1145, 1139, 1131, 1166, 1168, 1130,   30, 1217, 1162,
        114, 1143, 1215, 1186, 1142, 1163, 1184,  328, 1173, 1156, 1167,
       1158, 1133, 1005, 1183, 1155, 1170, 1164, 1182, 1175, 1154, 1148,
       1187,    6, 1172, 1180, 1234, 1129, 1149, 1137, 1185, 1200, 1159,
       1152, 1151, 1178, 1177, 1181, 1179, 1176, 1174, 1203,  108, 1128,
       1144, 1171,  763, 1157, 1224,  243,    7, 1141, 1135, 1228,  221,
        775, 1230, 1232, 1134, 1191,   20, 1229, 1231, 1227, 1235, 1225,
       1114, 1233, 1226,    8,  134, 1105,  141, 1169, 1126, 1127,  127,
       1107, 1124, 1199, 1218, 1122, 1125, 1121, 1118, 1123, 1116,  140,
       1117, 1120, 1108, 1112, 1119,  132, 1211,  115, 1216, 1106,   13,
       1093, 1084,  779, 1223,  767, 1194, 1081,  457, 1212,    1, 1096,
        787,  125,   24,   31, 1060, 1058, 1206,  113, 1076, 1078, 1056,
       1074, 1028, 1039,  232, 1047,  976, 1041, 1025, 1027, 1038, 1075,
       1079, 1077, 1052, 1070, 1034, 1071, 1068,  236, 1072, 1073, 1057,
       1069,  137,  244, 1023,  791, 1043, 1044, 1032, 1065,  352, 1051,
       1062, 1214,  128, 1054,  245, 1021,  554, 1050, 1030,  988, 1029,
       1066,  985, 1036, 1220,  225, 1059, 1049, 1031, 1064, 1104,  758,
        452, 1053,  220,  973,  987, 1055,  135, 1033, 1026, 1042, 1196,
        770, 1040, 1046, 1035, 1037, 1024, 1045, 1048,  247,  224, 1213,
        153, 1022, 1208, 1020,  542,  155,  683,  338,  237, 1111, 1063,
       1067,  242,  440,  882, 1061]

      block_num = 0

      for block_id in skip_block:
        layer_out = (int)(block_id / 108)
        layer_in = (int)(block_id % 108)
        block = 0
        layer = 0
        if layer_in < 48:
          layer = (int) (layer_in / 12)
          block = (int) (layer_in % 12)
        elif layer_in < 96:
          layer = 4
          block = layer_in - 48
        elif layer_in < 108:
          layer = 5
          block = layer_in - 96
        
        layer = layer_out * 6 + layer
        
        print(layer)
        print(block)

        layer_list = l2_norm_list[layer]
        block_list = layer_list[block]
        block_list[:] = 0
        

        block_num += 1
        if block_num >= block_remain:
          break

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

    if not prune_layer_wise and pruning_type == 5:
      # block wise pruning on whole network
      tf.logging.info("[INFO] block wise pruning on whole network with G={} and sparsity={}".format(granularity, sparsity))
      l2_norm_list = []
      transpose = False
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_size = granularity
        i = 0
        while i < score.shape[0]:
            j = 0
            while j < score.shape[1]:
                l2_norm_value = np.linalg.norm(score[i:i+block_size, j:j+block_size])
                l2_norm_list.append(l2_norm_value)
                j += block_size
            i += block_size
      
      threshold = np.percentile(l2_norm_list, sparsity)
      block_mask = l2_norm_list > threshold
      mask_index = 0
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        mask = np.zeros(score.shape)
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

      sess.run(assign_ops, feed)
      return


    if pruning_type == 13:      

      ###############
      ## To skip the Densor Blocks.
      ###############

      # Top-300 Blocks with G = 64
      skip_block = [1195, 1207, 1098, 1086, 1097, 1085, 1263, 1278, 1264, 1238, 1294,
       1268, 1250, 1284, 1262, 1276, 1288, 1292, 1295, 1290, 1242, 1274,
       1287, 1291, 1289, 1293, 1272, 1286, 1256, 1285, 1273, 1240, 1258,
       1253, 1244, 1261, 1239, 1281, 1246, 1283, 1270, 1241, 1251, 1279,
       1245, 1243, 1254, 1266, 1271, 1255, 1109, 1267, 1280, 1260, 1248,
       1265, 1257, 1110, 1236, 1237, 1269, 1249, 1282, 1252, 1247, 1259,
         16, 1193,   12, 1277, 1275, 1219,  445,  993,  138,  340,    0,
        433,   18, 1136, 1146, 1205, 1161,  981, 1132,  126,    4,  123,
       1188, 1165,  111, 1140,   19, 1160, 1153, 1138, 1147,  233,   28,
        120, 1150, 1145, 1139, 1131, 1166, 1168, 1130,   30, 1217, 1162,
        114, 1143, 1215, 1186, 1142, 1163, 1184,  328, 1173, 1156, 1167,
       1158, 1133, 1005, 1183, 1155, 1170, 1164, 1182, 1175, 1154, 1148,
       1187,    6, 1172, 1180, 1234, 1129, 1149, 1137, 1185, 1200, 1159,
       1152, 1151, 1178, 1177, 1181, 1179, 1176, 1174, 1203,  108, 1128,
       1144, 1171,  763, 1157, 1224,  243,    7, 1141, 1135, 1228,  221,
        775, 1230, 1232, 1134, 1191,   20, 1229, 1231, 1227, 1235, 1225,
       1114, 1233, 1226,    8,  134, 1105,  141, 1169, 1126, 1127,  127,
       1107, 1124, 1199, 1218, 1122, 1125, 1121, 1118, 1123, 1116,  140,
       1117, 1120, 1108, 1112, 1119,  132, 1211,  115, 1216, 1106,   13,
       1093, 1084,  779, 1223,  767, 1194, 1081,  457, 1212,    1, 1096,
        787,  125,   24,   31, 1060, 1058, 1206,  113, 1076, 1078, 1056,
       1074, 1028, 1039,  232, 1047,  976, 1041, 1025, 1027, 1038, 1075,
       1079, 1077, 1052, 1070, 1034, 1071, 1068,  236, 1072, 1073, 1057,
       1069,  137,  244, 1023,  791, 1043, 1044, 1032, 1065,  352, 1051,
       1062, 1214,  128, 1054,  245, 1021,  554, 1050, 1030,  988, 1029,
       1066,  985, 1036, 1220,  225, 1059, 1049, 1031, 1064, 1104,  758,
        452, 1053,  220,  973,  987, 1055,  135, 1033, 1026, 1042, 1196,
        770, 1040, 1046, 1035, 1037, 1024, 1045, 1048,  247,  224, 1213,
        153, 1022, 1208, 1020,  542,  155,  683,  338,  237, 1111, 1063,
       1067,  242,  440,  882, 1061]

      block_to_div_n = 0

      layer_array = np.zeros(block_remain)
      block_array = np.zeros(block_remain)

      lyaer_dim = (768 * (4 + 4 + 1)) / granularity
      block_dim = 768 / granularity

      for block_id in skip_block:
        if block_remain == 0:
          return
        layer_out = (int)(block_id / lyaer_dim)
        layer_in = (int)(block_id % lyaer_dim)
        block = 0
        layer = 0
        if layer_in < block_dim * 4:
          ## 0 - 3 layers
          layer = (int) (layer_in / block_dim)
          block = (int) (layer_in % block_dim)
        elif layer_in < block_dim * 8:
          ## 4 layer
          layer = 4
          block = layer_in - block_dim * 4
        elif layer_in < lyaer_dim:
          ## 5 layer
          layer = 5
          block = layer_in - block_dim * 8
        
        layer = layer_out * 6 + layer
        
        # layer_list = l2_norm_list[layer]
        # block_list = layer_list[block]
        # block_list[:] = np.inf
        layer_array[block_to_div_n] = layer
        block_array[block_to_div_n] = block

        block_to_div_n += 1
        if block_to_div_n >= block_remain:
          break
    
      layer_mask = np.zeros([len(accumulated_scores), int(block_dim * 4)])
      for block_id in range(block_remain):
        layer_mask[int(layer_array[block_id])][int(block_array[block_id])] = 1

      # granularity pruning on whole layer
      # AW = 1 and AW = granularity
      l2_norm_list = []
      block_size_list = []
      transpose = False
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        if transpose:
          score = score.T
        block_num = score.shape[1] / granularity

        block_to_div = 0
        for bid in range(int(block_num)):
          if layer_mask[layer][bid] == 1:
            block_to_div += 1
        
        print("block_to_div")
        print(block_to_div)
        block_to_div = block_to_div * granularity + block_num - block_to_div

        print(block_to_div)
        print(" ")


        block_split_array = np.zeros(int(block_to_div - 1), dtype=int)
        block_split_index = 0
        block_split_value = 0

        for bid in range(int(block_num)):
          if layer_mask[layer][bid] == 0:
            block_split_value += granularity
            block_split_array[block_split_index] = block_split_value
            block_split_index += 1
            if block_split_index == block_to_div - 1:
              break
          else:
            for ii in range(granularity):
              block_split_value += 1
              block_split_array[block_split_index] = int(block_split_value)
              block_split_index += 1
              if block_split_index == block_to_div - 1:
                break              
            if block_split_index == block_to_div - 1:
              break
      
        assert(block_split_index == block_to_div - 1)
        assert(block_split_value < score.shape[1])



        split_score = np.split(score, list(block_split_array), axis=1)
      
        print("split_score")
        print(len(split_score))
        print(block_to_div)
        print(" ")
        assert(len(split_score) == block_to_div)

        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        split_block_size = []#[np.concatenate([np.expand_dims(w.shape[1], axis=1) for i in range(w.shape[0])]) for w in split_score]
        for b_s in split_score:
          b_s_array = np.empty(b_s.shape[0])
          b_s_array[:] = b_s.shape[1]
          split_block_size.append(b_s_array)
          
        l2_norm_list.append(split_l2_norm)
        block_size_list.append(split_block_size)

        b_n = np.sum(split_block_size)

        print("b_n")
        print(b_n)
        print(" ")

        assert(b_n == score.shape[1] * score.shape[0])


      split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
      split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]

      split_layer_block_size_list = [ v for n in block_size_list for v in n]
      split_weight_block_size_list = [ v for n in split_layer_block_size_list for v in n]

      element_num = np.sum(split_weight_block_size_list)
      print("element_num")
      print(element_num)

      element_num = 0
      for ii in split_weight_block_size_list:
        assert(ii != np.nan)
        element_num += ii
      print("element_num")
      print(element_num)

      print((768 * 768) * (4 + 4 + 4) * 12)
      print(" ")
      
      assert(element_num == (768 * 768) * (4 + 4 + 4) * 12)

      pair_block = list(zip(split_weight_l2_norm_list, split_weight_block_size_list))
      pair_block = sorted(pair_block)

      print("sparsity")
      print(sparsity)
      threshold = 0
      ele_n = 0
      for pair in pair_block:
        ele_n += pair[1]
        if ele_n * 100 / element_num > sparsity:
          threshold = pair[0]
          break

      print("threshold")
      print(threshold)
      print("ele_n")
      print(ele_n)
      #threshold = np.percentile(split_weight_l2_norm_list, sparsity)

      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        if transpose:
          score = score.T
        block_num = score.shape[1] / granularity
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_size = block_size_list[layer]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(int(n[0]))], axis=1) for m,n in zip(split_mask, split_size) ]
        mask = np.concatenate(split_mask, axis = 1)
        if transpose:
          mask = mask.T
        assert(mask.shape == placeholders[layer].shape)
        feed[placeholders[layer]] = mask

      #assert(0)
      sess.run(assign_ops, feed)
      return

    if pruning_type == 11:
      # granularity pruning on whole layer
      # AW = 0 and AW = granularity
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
      

      ###############
      ## To skip the Densor Blocks.
      ###############

      # Top-300 Blocks with G = 64
      skip_block = [1090, 1095,  980,  992, 1102, 1083,  995,  435,  447,  336,  324,
        983,  459,  222,  660,  672,  648,  234,  990,  229,  572,  978,
       1099,  238, 1087,  217,  226,  557,  545,  119,  781,  131,  897,
        780,  571,  972,  984,  895,  544,  559,  885,  564,  677,  680,
        556,  109,  334,  461,  889, 1080,  569,  788, 1092,  460,  790,
        679,  121,  463,  346,  778,  567,  547,  441,  873,  899,  785,
        668,  887, 1088,  789,  766,  875,  451,  453,  865,  568,  877,
        656,  465,  333,  896,  218,  345, 1006,  673,  332, 1100,  439,
        658,  344, 1002,  893,  456,  449,  241,  579,  577,  655,  674,
        546,  676,  462,  558,  437,  580,  560, 1007, 1001,  357,  466,
        489, 1004,  678,  670,  230,  570,    2,  358,  629,  574, 1101,
         14,  373,  724,  381,  583,  584,  112, 1094,  687,  582,  481,
        768,  704,  391,  667,  587,  458,  552,  699,  756,  685,  511,
        586,  518,  471,  359,  502,  388,  469,  590,  578,  519,  581,
        514,  378,  769, 1082,   22,  594,  585,  504,  702,  607,  703,
        400,  522,  576,  731,  627,  486,  888,  520,  606,  419,  496,
        742,  874,  631,  735,  383,  688,  485,  691,  382,  621,  527,
        385,  124,  741,  513,  695,  472,  633,  351,  505,  614,  722,
        374,  476,  434, 1003,  376,  133, 1089,  690,  692,  733,  599,
        474,  483,  904,  397,  411,  604,  521,  605,  548,  710,  495,
        399,  738,  475,  609,  487,  136,  665,  597,  377,  632,  630,
        499,  898,  726,  488,  543,  903,  470,  329,  593,  686,  707,
        508,  478,  404,  491,  482,  479,  507,  402,  628,  384,  595,
        510,  523,  412,  602,  714,  901,  616,  694,   26,  689,  498,
        592,  526,  612,  492,  615,  407,  473,  379,  693,  624,  288,
        617,  684,  727,  417,  410,  503,  468,  500,  996,  413,  700,
        706,  623,  142,  393,  608,  600,  418,  310,  414,  622,  386,
        906,  299,  341]

      block_num = 0
      block_remain = 36

      for block_id in skip_block:
        layer_out = (int)(block_id / 108)
        layer_in = (int)(block_id % 108)
        block = 0
        layer = 0
        if layer_in < 48:
          layer = (int) (layer_in / 12)
          block = (int) (layer_in % 12)
        elif layer_in < 96:
          layer = 4
          block = layer_in - 48
        elif layer_in < 108:
          layer = 5
          block = layer_in - 96
        
        layer = layer_out * 6 + layer
        
        print(layer)
        print(block)

        layer_list = l2_norm_list[layer]
        block_list = layer_list[block]
        block_list[:] = np.inf
        

        block_num += 1
        if block_num >= block_remain:
          break

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

    if not prune_layer_wise and pruning_type == 5:
      # block wise pruning on whole network
      tf.logging.info("[INFO] block wise pruning on whole network with G={} and sparsity={}".format(granularity, sparsity))
      l2_norm_list = []
      transpose = False
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_size = granularity
        i = 0
        while i < score.shape[0]:
            j = 0
            while j < score.shape[1]:
                l2_norm_value = np.linalg.norm(score[i:i+block_size, j:j+block_size])
                l2_norm_list.append(l2_norm_value)
                j += block_size
            i += block_size
      
      threshold = np.percentile(l2_norm_list, sparsity)
      block_mask = l2_norm_list > threshold
      mask_index = 0
      for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        mask = np.zeros(score.shape)
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

      sess.run(assign_ops, feed)
      return

    if pruning_type == 7:
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
          # prune N + K dimension on whole network
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

    if pruning_type == 9:
      # prune N + K dimension on whole network
      # N dimension pruning whole column
      # K dimension pruning uses granularity pruning
      # prune N dimesnion first is necessary, and hence consider the pruning on K and N independently. 
      
      # pruning on N

      #global pruning on attention and fc seperatedly

      origin_accumulated_scores = accumulated_scores

      attention_mask = []
      accumulated_scores = [score for i,score in enumerate(origin_accumulated_scores) if i%6<3]
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
        # assert(tiled_mask[layer].shape == placeholders[layer].shape)
        # feed[placeholders[layer]] = mask * tiled_mask[layer]
        attention_mask.append(mask * tiled_mask[layer])

      fc_mask = []
      accumulated_scores = [score for i,score in enumerate(origin_accumulated_scores) if i%6>=3]
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
        # assert(tiled_mask[layer].shape == placeholders[layer].shape)
        # feed[placeholders[layer]] = mask * tiled_mask[layer]
        fc_mask.append(mask * tiled_mask[layer])

      mask = []
      for layer in range(len(origin_accumulated_scores)):
        if layer%6<3:
          mask.append(attention_mask.pop(0))
        if layer%6>=3:
          mask.append(fc_mask.pop(0))
      assert len(attention_mask)==0
      assert len(fc_mask)==0
      assert len(mask)==len(origin_accumulated_scores)
      for layer in range(len(origin_accumulated_scores)):
        assert placeholders[layer].shape == mask[layer].shape
        feed[placeholders[layer]] = mask[layer]

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

