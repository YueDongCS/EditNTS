#!/usr/bin/env python
# coding:utf8

from __future__ import print_function

import argparse
import collections
import logging

import numpy as np
import torch
import torch.nn as nn

import data
from checkpoint import Checkpoint
from editnts import EditNTS
from evaluator import Evaluator

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

def sort_by_lens(seq, seq_lengths):
    seq_lengths_sorted, sort_order = seq_lengths.sort(descending=True)
    seq_sorted = seq.index_select(0, sort_order)
    return seq_sorted, seq_lengths_sorted, sort_order

def reweigh_batch_loss(target_id_bath):
    pad_c = 0
    unk_c = 0
    keep_c = 0
    del_c = 0
    start_c = 0
    stop_c = 0
    other_c = 0

    new_edits_ids_l = target_id_bath
    for i in new_edits_ids_l:
        # start_c += 1
        # stop_c += 1
        for ed in i:
            if ed == PAD_ID:
                pad_c += 1
            elif ed == UNK_ID:
                unk_c += 1
            elif ed == KEEP_ID:
                keep_c += 1
            elif ed == DEL_ID:
                del_c += 1
            elif ed == START_ID:
                start_c +=1
            elif ed == STOP_ID:
                stop_c +=1
            else:
                other_c += 1

    NLL_weight = np.zeros(30006) + (1 / other_c+1)
    NLL_weight[PAD_ID] = 0  # pad
    NLL_weight[UNK_ID] = 1. / unk_c+1
    NLL_weight[KEEP_ID] = 1. / keep_c+1
    NLL_weight[DEL_ID] = 1. / del_c+1
    NLL_weight[5] = 1. / stop_c+1
    NLL_weight_t = torch.from_numpy(NLL_weight).float().cuda()
    # print(pad_c, unk_c, start_c, stop_c, keep_c, del_c, other_c)
    return NLL_weight_t

def reweight_global_loss(w_add,w_keep,w_del):
    # keep, del, other, (0, 65304, 246768, 246768, 2781648, 3847848, 2016880) pad,start,stop,keep,del,add
    NLL_weight = np.ones(30006)+w_add
    NLL_weight[PAD_ID] = 0  # pad
    NLL_weight[KEEP_ID] = w_keep
    NLL_weight[DEL_ID] = w_del
    return NLL_weight

def training(edit_net,nepochs, args, vocab, print_every=100, check_every=500):
    eval_dataset = data.Dataset(args.data_path + 'val.df.filtered.pos') # load eval dataset
    evaluator = Evaluator(loss= nn.NLLLoss(ignore_index=vocab.w2i['PAD'], reduction='none'))
    editnet_optimizer = torch.optim.Adam(edit_net.parameters(),
                                          lr=1e-3, weight_decay=1e-6)
    # scheduler = MultiStepLR(abstract_optimizer, milestones=[20,30,40], gamma=0.1)
    # abstract_scheduler = ReduceLROnPlateau(abstract_optimizer, mode='max')

    # uncomment this part to re-weight different operations
    # NLL_weight = reweight_global_loss(args.w_add, args.w_keep, args.w_del)
    # NLL_weight_t = torch.from_numpy(NLL_weight).float().cuda()
    # editnet_criterion = nn.NLLLoss(weight=NLL_weight_t, ignore_index=vocab.w2i['PAD'], reduce=False)
    editnet_criterion = nn.NLLLoss(ignore_index=vocab.w2i['PAD'], reduction='none')

    best_eval_loss = 0. # init statistics
    print_loss = []  # Reset every print_every

    for epoch in range(nepochs):
        # scheduler.step()
        #reload training for every epoch
        if os.path.isfile(args.data_path+'train.df.filtered.pos'):
            train_dataset = data.Dataset(args.data_path + 'train.df.filtered.pos')
        else:  # iter chunks and vocab_data
            train_dataset = data.Datachunk(args.data_path + 'train.df.filtered.pos')

        for i, batch_df in train_dataset.batch_generator(batch_size=args.batch_size, shuffle=True):

            #     time1 = time.time()
            prepared_batch, syn_tokens_list = data.prepare_batch(batch_df, vocab, args.max_seq_len) #comp,scpn,simp

            # a batch of complex tokens in vocab ids, sorted in descending order
            org_ids = prepared_batch[0]
            org_lens = org_ids.ne(0).sum(1)
            org = sort_by_lens(org_ids, org_lens)  # inp=[inp_sorted, inp_lengths_sorted, inp_sort_order]
            # a batch of pos-tags in pos-tag ids for complex
            org_pos_ids = prepared_batch[1]
            org_pos_lens = org_pos_ids.ne(0).sum(1)
            org_pos = sort_by_lens(org_pos_ids, org_pos_lens)

            out = prepared_batch[2][:, :]
            tar = prepared_batch[2][:, 1:]

            simp_ids = prepared_batch[3]

            editnet_optimizer.zero_grad()
            output = edit_net(org, out, org_ids, org_pos,simp_ids)
            ##################calculate loss
            tar_lens = tar.ne(0).sum(1).float()
            tar_flat=tar.contiguous().view(-1)
            loss = editnet_criterion(output.contiguous().view(-1, vocab.count), tar_flat).contiguous()
            loss[tar_flat == 1] = 0 #remove loss for UNK
            loss = loss.view(tar.size())
            loss = loss.sum(1).float()
            loss = loss/tar_lens
            loss = loss.mean()

            print_loss.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(edit_net.parameters(), 1.)
            editnet_optimizer.step()

            if i % print_every == 0:
                log_msg = 'Epoch: %d, Step: %d, Loss: %.4f' % (
                    epoch,i, np.mean(print_loss))
                print_loss = []
                print(log_msg)

                # Checkpoint
            if i % check_every == 0:
                edit_net.eval()

                val_loss, bleu_score, sari, sys_out = evaluator.evaluate(eval_dataset, vocab, edit_net,args)
                log_msg = "epoch %d, step %d, Dev loss: %.4f, Bleu score: %.4f, Sari: %.4f \n" % (epoch, i, val_loss, bleu_score, sari)
                print(log_msg)

                if val_loss < best_eval_loss:
                    best_eval_loss = val_loss
                Checkpoint(model=edit_net,
                           opt=editnet_optimizer,
                           epoch=epoch, step=i,
                           ).save(args.store_dir)
                print("checked after %d steps"%i)

                edit_net.train()
    return edit_net

dataset='newsela'
def main():
    torch.manual_seed(233)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,dest='data_path',
                        default='/home/ml/ydong26/data/EditNTS_data/editnet_data/%s/'%dataset,
                        help='Path to train vocab_data')
    parser.add_argument('--store_dir', action='store', dest='store_dir',
                        default='/home/ml/ydong26/tmp_store/editNTS_%s'%dataset,
                        help='Path to exp storage directory.')
    parser.add_argument('--vocab_path', type=str, dest='vocab_path',
                        default='../vocab_data/',
                        help='Path contains vocab, embedding, postag_set')
    parser.add_argument('--load_model', type=str, dest='load_model',
                        default=None,
                        help='Path for loading pre-trained model for further training')

    parser.add_argument('--vocab_size', dest='vocab_size', default=30000, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--max_seq_len', dest='max_seq_len', default=100)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=int, default=1,
                        help='select GPU')

    #train_file = '/media/vocab_data/yue/TS/editnet_data/%s/train.df.filtered.pos'%dataset
    # test='/media/vocab_data/yue/TS/editnet_data/%s/test.df.pos' % args.dataset
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

            # load vocab-related files and init vocab
    print('*'*10)
    vocab = data.Vocab()
    vocab.add_vocab_from_file(args.vocab_path+'vocab.txt', args.vocab_size)
    vocab.add_embedding(gloveFile=args.vocab_path+'glove.6B.100d.txt')
    pos_vocab = data.POSvocab(args.vocab_path) #load pos-tags embeddings
    print('*' * 10)

    print(args)
    print("generating config")
    hyperparams=collections.namedtuple(
        'hps', #hyper=parameters
        ['vocab_size', 'embedding_dim',
         'word_hidden_units', 'sent_hidden_units',
         'pretrained_embedding', 'word2id', 'id2word',
         'pos_vocab_size', 'pos_embedding_dim']
    )
    hps = hyperparams(
        vocab_size=vocab.count,
        embedding_dim=100,
        word_hidden_units=args.hidden,
        sent_hidden_units=args.hidden,
        pretrained_embedding=vocab.embedding,
        word2id=vocab.w2i,
        id2word=vocab.i2w,
        pos_vocab_size=pos_vocab.count,
        pos_embedding_dim=30
    )

    print('init editNTS model')
    edit_net = EditNTS(hps, n_layers=1)
    edit_net.cuda()

    if args.load_model is not None:
        print("load edit_net for further training")
        ckpt_path = args.load_model
        ckpt = Checkpoint.load(ckpt_path)
        edit_net = ckpt.model
        edit_net.cuda()
        edit_net.train()

    training(edit_net, args.epochs, args, vocab)


if __name__ == '__main__':
    import os
    cwd = os.getcwd()
    print(cwd)

    main()
