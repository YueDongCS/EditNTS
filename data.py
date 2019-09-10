from collections import Counter

import glob
import random
import struct
import csv
import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import random
import pickle
# <s> and </s> are used in the vocab_data files to segment the abstracts into sentences. They don't receive vocab ids.


PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

PAD_ID = 0 #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK_ID = 1 #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP_ID = 2 # This has a vocab id, which is used for copying from the source [2]
DEL_ID = 3 # This has a vocab id, which is used for deleting the corresponding word [3]
START_ID = 4 # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP_ID = 5 # This has a vocab id, which is used to stop decoding [5]

def sent2id(sent,vocab):
    """
    this function transfers a sentence (in list of strings) to an np_array
    :param sent: sentence in list of strings
    :param vocab: vocab object
    :return: sentence in numeric token numbers
    """
    new_sent = np.array([[vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in sent]])
    return new_sent

def id2edits(ids,vocab):
    """
    #     this function transfers a id sentences of edits to actual edit actions
    #     :param ids: list of ids indicating edits
    #     :param vocab: vocab object
    #     :return: list of actual edits
    #     """
    edit_list = [vocab.i2w[i] for i in ids]
    return edit_list


def batchify(data, max_len=100): #max_len cutout defined by human
    bsz = len(data)
    try:
        maxlen_data = max([s.shape[0] for s in data])
    except:
        maxlen_data = max([len(s) for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        try:
            batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        except:
            batch[i, :min(len(s), maxlen)] = s[:min(len(s), maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def batchify_start_stop(data, max_len=100,start_id=4,stop_id=5): #max_len cutout defined by human
    # add start token at the beginning and stop token at the end of each sequence in a batch
    data = [np.append(s, [stop_id]) for s in data]  # stop 3
    data = [np.insert(s, 0, start_id) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def batchify_stop(data, max_len=100,start_id=4,stop_id=5): #max_len cutout defined by human
    # add stop tokens at the end of the sequence in each batch
    data = [np.append(s, [stop_id]) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()

class Vocab():
    def __init__(self):
        self.word_list = [PAD, UNK, KEEP, DEL, START, STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def add_vocab_from_file(self, vocab_file="../vocab_data/vocab.txt",vocab_size=30000):
        with open(vocab_file, "rb") as f:
            for i,line in enumerate(f):
                if i >=vocab_size:
                    break
                self.word_list.append(line.split()[0])  # only want the word, not the count
        print("read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile="path_for_glove_embedding", embed_size=100):
        print("Loading Glove embeddings")
        with open(gloveFile, 'r') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    # if len(model) % 1000 == 0:
                        # print("processed %d vocab_data" % len(model))
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))

class POSvocab():
    def __init__(self,vocab_path):
        self.word_list = [PAD,UNK,START,STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None
        with open(vocab_path+'postag_set.p','r') as f:
            # postag_set is from NLTK
            tagdict = pickle.load(f)

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

        for w in tagdict:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

class Datachunk():
    def __init__(self,data_path):
        self.data_path = data_path
        self.listdir = os.listdir(self.data_path)
        random.shuffle(self.listdir)
        self.idx_count = 0

    def example_generator(self,shuffle=True):
        while len(self.listdir) != 0:
            print("reading a new chunk with %d chunks remaining" % len(self.listdir))
            df = pd.read_pickle(self.data_path + self.listdir.pop())

            if shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
                print('shuffling the df')

            for index, row in df.iterrows():
                self.idx_count+=1
                yield self.idx_count, row

    def batch_generator(self, batch_size=1, shuffle=True):
        while len(self.listdir) != 0:
            # print("reading a new chunk with %d chunks remaining" % len(self.listdir))
            df = pd.read_pickle(self.data_path + self.listdir.pop())
            if shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
                # print('shuffling the df')

            list_df = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
            for df in list_df:
                self.idx_count += 1
                yield self.idx_count, df

class Dataset():
    def __init__(self,data_path):
        self.df = pd.read_pickle(data_path)
        self.idx_count = 0

    def example_generator(self):
        for index, row in self.df.iterrows():
            yield index, row

    def batch_generator(self, batch_size=64, shuffle=True):
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            # print('shuffling the df')

        list_df = [self.df[i:i + batch_size] for i in range(0, self.df.shape[0], batch_size)]
        for df in list_df:
            self.idx_count += 1
            yield self.idx_count, df


def prepare_batch(batch_df,vocab, max_length=100):
    """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_ids', edit_labels','new_edit_ids']
        :param vocab: vocab object for translation
        :return: inp: original input sentences
        :return: inp_pos: pos-tag ids for the input sentences
        :return: tgt: the target edit-labels in ids
        :return: inp_simp:the corresponding simple sentences in ids
        :return: batch_df['comp_tokens']:the complex tokens
        """
    inp = batchify_stop(batch_df['comp_ids'], max_len=max_length)
    inp_pos = batchify_stop(batch_df['comp_pos_ids'], max_len=max_length)
    inp_simp=batchify_start_stop(batch_df['simp_id'], max_len=max_length)
    # tgt = batchify_start_stop(batch_df['edit_ids'], max_len=max_length)  # edit ids has early stop
    tgt = batchify_start_stop(batch_df['new_edit_ids'], max_len=max_length)  # new_edit_ids do not do early stopping
    # I think new edit ids do not ave early stopping
    return [inp, inp_pos, tgt,inp_simp], batch_df['comp_tokens']





