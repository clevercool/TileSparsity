import os
import pickle
import numpy as np


def vector_wise(score):
    block_size = 16
    sparsity_hist = np.zeros(block_size+1)
    for i in range(score.shape[0]):
        j = 0
        while j < score.shape[1]:
            sparsity_hist[np.sum(score[i][j:j+block_size])] +=1
            j += block_size
    sparsity_hist /= np.sum(sparsity_hist)
    return sparsity_hist

def block_wise(score):
    block_size = 8
    sparsity_hist = np.zeros(block_size*block_size+1)
    i = 0
    while i < score.shape[0]:
        j = 0
        while j < score.shape[1]:
            sparsity_hist[np.sum(score[i:i+block_size, j:j+block_size])] += 1
            print(np.sum(score[i:i+block_size, j:j+block_size]))
            j += block_size
        i += block_size
    sparsity_hist /= np.sum(sparsity_hist)
    return sparsity_hist    

def aligned_wise(score):
    block_size = 64
    sparsity_hist = np.zeros(block_size+1)
    i = 0
    while i < score.shape[0]:
        j = 0
        while j < score.shape[1]:
            sparsity_hist[np.sum(score[i, j:j+block_size])] += 1
            print(np.sum(score[i, j:j+block_size]))
            j += block_size
        i += 1
    sparsity_hist /= np.sum(sparsity_hist)
    return sparsity_hist  

if __name__ =="__main__":

    with  open('./profile/model.ckpt-73631_weight_fc.pickle', 'rb') as file:
        weights = pickle.load(file)   


    threshold = np.percentile(weights[0][0], 75)
    score = weights[0][0]>threshold
    hist = aligned_wise(score) 
