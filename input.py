from collections import defaultdict
from os import listdir
from sklearn.model_selection import train_test_split
from model import create_model
from vocabulary import load_vocabulary
import numpy as np
from numpy.random import choice, shuffle
import pickle
import keras
from itertools import chain

vocabulary = load_vocabulary()
N = 1024

def chunk(arr_w, arr_l):
    # split array into two subarrays [1...n][n+1...k]
    # combine lists into [n+1...k][1...n]
    start_index = choice(range(len(arr_l)), 1)[0]
    pre_arr_w, post_arr_w = arr_w[:start_index], arr_w[start_index:]
    pre_arr_l, post_arr_l = arr_l[:start_index], arr_l[start_index:]

    # iterate over [n+1...k][1...n] and create chunks of size N
    chunks_w, chunks_l = [], []
    subw, subl = [], []
    for w, l in zip(chain(post_arr_w, pre_arr_w), chain(post_arr_l, pre_arr_l)):
        subw.append(w)
        subl.append(l)
        if len(subl) == N:
            chunks_w.append(np.array(subw))
            chunks_l.append(np.array(subl))
            subw, subl = [], []

    # if last chunk < N, take items from [n+1...k] until chunk is 1024 items, then return all
    if len(subl) < N:
        for w, l in zip(post_arr_w, post_arr_l):
            subw.append(w)
            subl.append(l)
            if len(subl) == N:
                chunks_w.append(np.array(subw))
                chunks_l.append(np.array(subl))
                return np.array(chunks_w), np.array(chunks_l)
    else:
        return np.array(chunks_w), np.array(chunks_l)

def re_chunk(indices, labels):
    new_indices, new_labels = [], []

    for ind, lbl in zip(indices, labels):
        ind, lbl = chunk(ind.flatten(), lbl.flatten())
        new_indices.append(ind)
        new_labels.append(lbl)
    
    return np.array(new_indices), np.array(new_labels)

def format_book(data):
    # translate words into integers from vocabulary
    indices = np.array(list(map(lambda x: vocabulary[x], [x[0] for x in data])))
    labels = np.array(list(map(lambda x: int(x), [x[1] for x in data])))
    return chunk(indices, labels)

def read_book(bpath):
    with open(bpath) as f:
        return format_book([x.split('\t') for x in f.read().split('\n') if x][200:])

def train_val_split(data_folder):
    filenames = dict(enumerate(listdir(data_folder)))
    file_ids = np.array(range(len(filenames)))
    training = choice(file_ids, int(len(filenames)*0.5), replace=False)
    
    train = [data_folder+v for (k, v) in filenames.items() if k in training]
    val = [data_folder+v for (k, v) in filenames.items() if not k in training]
    
    return train, val

def batch_generator(indices, labels, reorder=True):
    batches = 32
    while True:
        for i in range(batches):
            yield np.stack((x[i] for x in indices), axis=0), np.stack((x[i] for x in labels), axis=0)
        if reorder:
            # re chunk data
            indices, labels = re_chunk(indices, labels)

def get_batch_generators():    
    data_folder = '/home/adam/git/nn_dialogue_structure/gutenberg_data/'

    train, val = train_val_split(data_folder)

    trainX_data = [read_book(x)[0] for x in train]
    trainY_data = [read_book(x)[1] for x in train]

    valX_data = [read_book(x)[0] for x in val]
    valY_data = [read_book(x)[1] for x in val]

    train_generator = batch_generator(trainX_data, trainY_data)
    val_generator = batch_generator(valX_data, valY_data)

    # for i, (x, y) in enumerate(val_generator):
    #     print(x.shape)
    #     print(y.shape)
    #     print()
    #     if i > 5:
    #         return 0

    return train_generator, val_generator

if __name__ == '__main__':
    get_batch_generators()
    