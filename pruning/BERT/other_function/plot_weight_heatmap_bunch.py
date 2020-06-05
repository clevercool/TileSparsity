#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:35:48 2019

@author: chandler
"""

import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./utils')
import modeling as  modeling
font_size=22

plot_range = 768
fig_size = 18

# 0 for wq, 1 wk, 2 wv
mat_idx = 1
weight_mat_name=['wq','wk','wv']




fig = plt.figure(figsize=(fig_size,fig_size))


ax = fig.add_subplot(2,2,1)
granularity =1

file_name = f'../profile/wqkv_layer0_granularity_{granularity}_pruned.npy'
fig_name = f'../fig/{weight_mat_name[mat_idx]}_layer0_first{plot_range}_granularity{granularity}_pruned_grey.png'

w_qkv = np.load(file_name,allow_pickle=True)

pruned = np.ma.masked_where(w_qkv[mat_idx+3][:plot_range,:plot_range],w_qkv[mat_idx][:plot_range,:plot_range])
remained = np.ma.masked_where(np.logical_not(w_qkv[mat_idx+3][:plot_range,:plot_range]),w_qkv[mat_idx][:plot_range,:plot_range])

#cax = ax.matshow(pruned,cmap=plt.get_cmap('Greys'),interpolation='None')
cax = ax.matshow(remained,cmap=plt.get_cmap('seismic'),interpolation='None')


plt.xticks(range(0,769,64),fontsize=font_size)
plt.yticks(range(0,769,64),fontsize=font_size)

ax.set_title(f'Granularity G = {granularity}',fontdict={'fontsize':font_size})


ax = fig.add_subplot(2,2,2)
granularity =128

file_name = f'../profile/wqkv_layer0_granularity_{granularity}_pruned.npy'
fig_name = f'../fig/{weight_mat_name[mat_idx]}_layer0_first{plot_range}_granularity{granularity}_pruned_grey.png'

w_qkv = np.load(file_name,allow_pickle=True)

pruned = np.ma.masked_where(w_qkv[mat_idx+3][:plot_range,:plot_range],w_qkv[mat_idx][:plot_range,:plot_range])
remained = np.ma.masked_where(np.logical_not(w_qkv[mat_idx+3][:plot_range,:plot_range]),w_qkv[mat_idx][:plot_range,:plot_range])

#cax = ax.matshow(pruned,cmap=plt.get_cmap('Greys'),interpolation='None')
cax = ax.matshow(remained,cmap=plt.get_cmap('seismic'),interpolation='None')


plt.xticks(range(0,769,64),fontsize=font_size)
plt.yticks(range(0,769,64),fontsize=font_size)

ax.set_title(f'Granularity G = {granularity}',fontdict={'fontsize':font_size})


ax = fig.add_subplot(2,2,3)
granularity =64

file_name = f'../profile/wqkv_layer0_granularity_{granularity}_pruned.npy'
fig_name = f'../fig/{weight_mat_name[mat_idx]}_layer0_first{plot_range}_granularity{granularity}_pruned_grey.png'

w_qkv = np.load(file_name,allow_pickle=True)

pruned = np.ma.masked_where(w_qkv[mat_idx+3][:plot_range,:plot_range],w_qkv[mat_idx][:plot_range,:plot_range])
remained = np.ma.masked_where(np.logical_not(w_qkv[mat_idx+3][:plot_range,:plot_range]),w_qkv[mat_idx][:plot_range,:plot_range])

#cax = ax.matshow(pruned,cmap=plt.get_cmap('Greys'),interpolation='None')
cax = ax.matshow(remained,cmap=plt.get_cmap('seismic'),interpolation='None')


plt.xticks(range(0,769,64),fontsize=font_size)
plt.yticks(range(0,769,64),fontsize=font_size)

ax.set_title(f'Granularity G = {granularity}',fontdict={'fontsize':font_size})




ax = fig.add_subplot(2,2,4)
granularity=768

file_name = f'../profile/wqkv_layer0_granularity_{granularity}_pruned.npy'
fig_name = f'../fig/{weight_mat_name[mat_idx]}_layer0_first{plot_range}_granularity{granularity}_pruned_grey.png'

w_qkv = np.load(file_name,allow_pickle=True)

pruned = np.ma.masked_where(w_qkv[mat_idx+3][:plot_range,:plot_range],w_qkv[mat_idx][:plot_range,:plot_range])
remained = np.ma.masked_where(np.logical_not(w_qkv[mat_idx+3][:plot_range,:plot_range]),w_qkv[mat_idx][:plot_range,:plot_range])

#cax = ax.matshow(pruned,cmap=plt.get_cmap('Blues'),interpolation='None')
cax = ax.matshow(remained,cmap=plt.get_cmap('seismic'),interpolation='None')


plt.xticks(range(0,769,64),fontsize=font_size)
plt.yticks(range(0,769,64),fontsize=font_size)

ax.set_title(f'Granularity G = {granularity}',fontdict={'fontsize':font_size})




###########naive plot
#w_qkv = np.load('../profile/wqkv_layer11.npy',allow_pickle=True)
#cax = ax.matshow(w_qkv[0][:plot_range,:plot_range],cmap=plt.get_cmap('hot_r'),interpolation='None')
 



############plot log tylor score
#w_qkv = np.load('/home/chandler/score.npy',allow_pickle=True)

#interest_area = w_qkv[0][:plot_range,:plot_range]
#interest_area = np.log(interest_area)
#cax = ax.matshow(interest_area,cmap=plt.get_cmap(''),interpolation='None')




##########plot with masked color

cb_ax = fig.add_axes([1, 0.2, 0.02, 0.6])
cb = fig.colorbar(cax, cax=cb_ax)
#cb = plt.colorbar(fig,fraction=0.035)
#cb = plt.colorbar(cax, ax=fig, shrink=0.95)
cb.ax.tick_params(labelsize=font_size)

plt.tight_layout()
plt.savefig('../fig/bunch.png',format='png',bbox_inches='tight')  

plt.show()
plt.close()

