from __future__ import unicode_literals, print_function, division

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
import data
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

MAX_LEN =100

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

def unsort(x_sorted, sorted_order):
    x_unsort = torch.zeros_like(x_sorted)
    x_unsort[:, sorted_order,:] = x_sorted
    return x_unsort

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pos_vocab_size, pos_embedding_dim,hidden_size, n_layers=1, embedding=None, embeddingPOS=None,dropout=0.3):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding

        if embeddingPOS is None:
            self.embeddingPOS = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        else:
            self.embeddingPOS = embeddingPOS

        self.rnn = nn.LSTM(embedding_dim+pos_embedding_dim, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, inp, inp_pos, hidden):
        #inp and inp pose should be both sorted
        inp_sorted=inp[0]
        inp_lengths_sorted=inp[1]
        inp_sort_order=inp[2]

        inp_pos_sorted = inp_pos[0]
        inp_pos_lengths_sorted = inp_pos[1]
        inp_pos_sort_order = inp_pos[2]

        emb = self.embedding(inp_sorted)
        emb_pos = self.embeddingPOS(inp_pos_sorted)

        embed_cat = torch.cat((emb,emb_pos),dim=2)
        packed_emb = pack(embed_cat, inp_lengths_sorted,batch_first=True)
        memory_bank, encoder_final = self.rnn(packed_emb, hidden)

        memory_bank = unpack(memory_bank)[0]
        memory_bank = unsort(memory_bank, inp_sort_order)

        h_unsorted=unsort(encoder_final[0], inp_sort_order)
        c_unsorted=unsort(encoder_final[1], inp_sort_order)


        return memory_bank.transpose(0,1), (h_unsorted,c_unsorted)

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers * 2, bsz, self.hidden_size).zero_()), \
               Variable(weight.new(self.n_layers * 2, bsz, self.hidden_size).zero_())


class EditDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers=1, embedding=None):
        super(EditDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding
        self.rnn_edits = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.rnn_words = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.attn_Projection_org = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.attn_Projection_scpn = nn.Linear(hidden_size, hidden_size, bias=False) #hard attention here


        self.attn_MLP = nn.Sequential(nn.Linear(hidden_size * 4, embedding_dim),
                                          nn.Tanh())
        self.out = nn.Linear(embedding_dim, self.vocab_size)
        self.out.weight.data = self.embedding.weight.data[:self.vocab_size]

    def execute(self, symbol, input, lm_state):
        """
        :param symbol: token_id for predicted edit action (in teacher forcing mode, give the true one)
        :param input: the word_id being editted currently
        :param lm_state: last lstm state
        :return:
        """
        # predicted_symbol = KEEP -> feed input to RNN_LM
        # predicted_symbol = DEL -> do nothing, return current lm_state
        # predicted_symbol = new word -> feed that word to RNN_LM
        is_keep = torch.eq(symbol, data.KEEP_ID)
        is_del = torch.eq(symbol, data.DEL_ID)
        if is_del:
            return lm_state
        elif is_keep: # return lstm with kept word learned in lstm
            _, new_lm_state = self.rnn_words(self.embedding(input.view(-1, 1)), lm_state)
        else: #consider as insert here
            # print(symbol.item())
            input = self.embedding(symbol.view(-1,1))
            _, new_lm_state = self.rnn_words(input,lm_state)
        return new_lm_state

    def execute_batch(self, batch_symbol, batch_input, batch_lm_state):
        batch_h = batch_lm_state[0]
        batch_c = batch_lm_state[1]

        bsz = batch_symbol.size(0)
        unbind_new_h = []
        unbind_new_c = []

        # unbind all batch inputs
        unbind_symbol = torch.unbind(batch_symbol,dim=0)
        unbind_input = torch.unbind(batch_input,dim=0)
        unbind_h = torch.unbind(batch_h,dim=1)
        unbind_c = torch.unbind(batch_c,dim=1)
        for i in range(bsz):
            elem=self.execute(unbind_symbol[i], unbind_input[i], (unbind_h[i].view(1,1,-1), unbind_c[i].view(1,1,-1)))
            unbind_new_h.append(elem[0])
            unbind_new_c.append(elem[1])
        new_batch_lm_h = torch.cat(unbind_new_h,dim=1)
        new_batch_lm_c = torch.cat(unbind_new_c,dim=1)
        return (new_batch_lm_h,new_batch_lm_c)


    def forward(self, input_edits, hidden_org,encoder_outputs_org, org_ids, simp_sent,teacher_forcing_ratio=1.):
        #input_edits and simp_sent need to be padded with START
        bsz, nsteps = input_edits.size()

        # revisit each word and then decide the action, for each action, do the modification and calculate rouge difference
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_out = []
        counter_for_keep_del = np.zeros(bsz, dtype=int)
        counter_for_keep_ins =np.zeros(bsz, dtype=int)
        # decoder in the training:

        if use_teacher_forcing:
            embedded_edits = self.embedding(input_edits)
            output_edits, hidden_edits = self.rnn_edits(embedded_edits, hidden_org)

            embedded_words = self.embedding(simp_sent)
            output_words, hidden_words = self.rnn_words(embedded_words, hidden_org)


            key_org = self.attn_Projection_org(output_edits)  # bsz x nsteps x nhid MIGHT USE WORD HERE
            logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
            attn_weights_org = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
            attn_applied_org = torch.bmm(attn_weights_org, encoder_outputs_org)  # bsz x nsteps x nhid

            for t in range(nsteps-1):
                # print(t)
                decoder_output_t = output_edits[:, t:t + 1, :]
                attn_applied_org_t = attn_applied_org[:, t:t + 1, :]

                ## find current word
                inds = torch.LongTensor(counter_for_keep_del)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c = encoder_outputs_org.gather(1, dummy)

                inds = torch.LongTensor(counter_for_keep_ins)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), output_words.size(2)).cuda()
                c_word = output_words.gather(1, dummy)

                output_t = torch.cat((decoder_output_t, attn_applied_org_t, c,c_word),
                                     2)  # bsz*nsteps x nhid*2
                output_t = self.attn_MLP(output_t)
                output_t = F.log_softmax(self.out(output_t), dim=-1)
                decoder_out.append(output_t)


                # interpreter's output from lm
                gold_action = input_edits[:, t + 1].data.cpu().numpy()  # might need to realign here because start added
                counter_for_keep_del = [i[0] + 1 if i[1] == 2 or i[1] == 3  else i[0]
                                        for i in zip(counter_for_keep_del, gold_action)]
                counter_for_keep_ins = [i[0] + 1 if i[1] != DEL_ID and i[1] != STOP_ID and i[1] != PAD_ID else i[0]
                                        for i in zip(counter_for_keep_ins, gold_action)]

                check1 = sum([x >= org_ids.size(1) for x in counter_for_keep_del])
                check2 = sum([x >= simp_sent.size(1) for x in counter_for_keep_ins])
                if check1 or check2:
                    # print(org_ids.size(1))
                    # print(counter_for_keep_del)
                    break


        else: # no teacher forcing
            decoder_input_edit = input_edits[:, :1]
            decoder_input_word=simp_sent[:,:1]
            t, tt = 0, max(MAX_LEN,input_edits.size(1)-1)

            # initialize
            embedded_edits = self.embedding(decoder_input_edit)
            output_edits, hidden_edits = self.rnn_edits(embedded_edits, hidden_org)

            embedded_words = self.embedding(decoder_input_word)
            output_words, hidden_words = self.rnn_words(embedded_words, hidden_org)
            #
            # # give previous word from tgt simp_sent
            # inds = torch.LongTensor(counter_for_keep_ins)
            # dummy = inds.view(-1, 1, 1)
            # dummy = dummy.expand(dummy.size(0), dummy.size(1), output_words.size(2)).cuda()
            # c_word = output_words.gather(1, dummy)

            while t < tt:
                if t>0:
                    embedded_edits = self.embedding(decoder_input_edit)
                    output_edits, hidden_edits = self.rnn_edits(embedded_edits, hidden_edits)

                key_org = self.attn_Projection_org(output_edits)  # bsz x nsteps x nhid
                logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
                attn_weights_org_t = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
                attn_applied_org_t = torch.bmm(attn_weights_org_t, encoder_outputs_org)  # bsz x nsteps x nhid

                ## find current word
                inds = torch.LongTensor(counter_for_keep_del)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c = encoder_outputs_org.gather(1, dummy)

                output_t = torch.cat((output_edits, attn_applied_org_t, c, hidden_words[0]),
                                     2)  # bsz*nsteps x nhid*2
                output_t = self.attn_MLP(output_t)
                output_t = F.log_softmax(self.out(output_t), dim=-1)

                decoder_out.append(output_t)
                decoder_input_edit=torch.argmax(output_t,dim=2)



                # gold_action = input[:, t + 1].vocab_data.cpu().numpy()  # might need to realign here because start added
                pred_action= torch.argmax(output_t,dim=2)
                counter_for_keep_del = [i[0] + 1 if i[1] == 2 or i[1] == 3 or i[1] == 5 else i[0]
                                        for i in zip(counter_for_keep_del, pred_action)]

                # update rnn_words
                # find previous generated word
                # give previous word from tgt simp_sent
                dummy_2 = inds.view(-1, 1).cuda()
                org_t = org_ids.gather(1, dummy_2)
                hidden_words = self.execute_batch(pred_action, org_t, hidden_words)  # we give the editted subsequence
                # hidden_words = self.execute_batch(pred_action, org_t, hidden_org)  #here we only give the word

                t += 1
                check = sum([x >= org_ids.size(1) for x in counter_for_keep_del])
                if check:
                    break
        return torch.cat(decoder_out, dim=1), hidden_edits

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()), \
               Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_())

    def beam_forwad_step(self,decoder_input_edits,hidden_edits,hidden_words, org_ids,encoder_outputs_org,counter_for_keep_del,beam_size=5):
        #buffers: each with k elements for next step
        decoder_input_k=[]
        hidden_edits_k=[]
        counter_for_keep_del_k=[]
        prob_k=[]
        hidden_words_k=[]

        # given decoder hidden, forward one step
        embedded = self.embedding(decoder_input_edits)
        decoder_output_t, hidden_edits = self.rnn_edits(embedded, hidden_edits)

        key_org = self.attn_Projection_org(decoder_output_t)  # bsz x nsteps x nhid
        logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
        attn_weights_org_t = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
        attn_applied_org_t = torch.bmm(attn_weights_org_t, encoder_outputs_org)  # bsz x nsteps x nhid

        ## find current word
        inds = torch.LongTensor(counter_for_keep_del)
        dummy = inds.view(-1, 1, 1)
        dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
        c = encoder_outputs_org.gather(1, dummy)

        output_t = torch.cat((decoder_output_t, attn_applied_org_t, c, hidden_words[0]),
                             2)  # bsz*nsteps x nhid*2
        output_t = self.attn_MLP(output_t)
        output_t = F.log_softmax(self.out(output_t), dim=-1)

        # update rnn_words
        # find previous generated word
        # give previous word from tgt simp_sent
        topv, topi = torch.topk(output_t,beam_size, dim=2)
        for b in range(beam_size):
            prob_t_k=topv[:,:,b]
            out_id_t_k=topi[:,:,b]
            counter_for_keep_del = [i[0] + 1 if i[1] == 2 or i[1] == 3 or i[1] == 5 else i[0]
                                    for i in zip(counter_for_keep_del, out_id_t_k)]

            dummy_2 = inds.view(-1, 1).cuda()
            org_t = org_ids.gather(1, dummy_2)
            hidden_words = self.execute_batch(out_id_t_k, org_t, hidden_words)  # input[:, t + 1]=gold action,

            decoder_input_k.append(out_id_t_k)
            hidden_edits_k.append(hidden_edits)
            prob_k.append(prob_t_k)
            hidden_words_k.append(hidden_words)
            counter_for_keep_del_k.append(counter_for_keep_del)
        return decoder_input_k,hidden_edits_k,hidden_words_k,prob_k,counter_for_keep_del_k



    def beam_forward(self, input_edits, simp_sent, hidden_org,encoder_outputs_org, org_ids, beam_size=5):
        # initialize for beam search
        bsz, nsteps = input_edits.size()
        # decoder_out = []
        counter_for_keep_del = np.zeros(bsz, dtype=int)
        # decoder_input = input[:, :1]
        t, tt = 0, max(MAX_LEN, input_edits.size(1) - 1)
        # embedded = self.embedding(decoder_input)
        # output, hidden = self.rnn(embedded, hidden_org)

        # initialize for beam list
        best_k_seqs = [[input_edits[:, :1]]]
        best_k_probs = [0.]
        best_k_hidden_edits = [hidden_org]
        best_k_hidden_words=[hidden_org]
        best_k_counters =[counter_for_keep_del]

        while t < tt:
            # print(t)
            next_best_k_squared_seq = []
            next_best_k_squared_probs = []
            next_best_k_squared_counters = []
            next_best_k_squared_hidden_edits = []
            next_best_k_squared_hidden_words = []

            for b in range(len(best_k_seqs)):
                seq = best_k_seqs[b]
                prob = best_k_probs[b]
                counter = best_k_counters[b]
                hidden_edits = best_k_hidden_edits[b]
                hidden_words = best_k_hidden_words[b]
                check = sum([x >= org_ids.size(1) for x in counter])
                if seq[-1].item() == STOP_ID or check:
                    # if end of token, make sure no children
                    next_best_k_squared_seq.append(seq)
                    next_best_k_squared_probs.append(prob)
                    next_best_k_squared_counters.append(counter)
                    next_best_k_squared_hidden_edits.append(hidden_edits)
                    next_best_k_squared_hidden_words.append(hidden_words)
                else:
                    # append the top k children
                    decoder_input_k, hidden_edits_k,hidden_words_k, prob_k, counter_for_keep_del_k=self.beam_forwad_step(seq[-1],
                                                         hidden_edits,hidden_words,org_ids,encoder_outputs_org,counter,beam_size)
                    for i in range(beam_size):
                        next_seq = seq[:]
                        next_seq.append(decoder_input_k[i])
                        next_best_k_squared_seq.append(next_seq)
                        next_best_k_squared_probs.append(prob + prob_k[i].item())
                        next_best_k_squared_counters.append(counter_for_keep_del_k[i])
                        next_best_k_squared_hidden_edits.append(hidden_edits_k[i])
                        next_best_k_squared_hidden_words.append(hidden_words_k[i])
            # contract to the best k
            indexs = np.argsort(next_best_k_squared_probs)[::-1][:beam_size]
            best_k_seqs = [next_best_k_squared_seq[i] for i in indexs]
            best_k_probs = [next_best_k_squared_probs[i] for i in indexs]
            best_k_counters = [next_best_k_squared_counters[i] for i in indexs]
            best_k_hidden_edits = [next_best_k_squared_hidden_edits[i] for i in indexs]
            best_k_hidden_words = [next_best_k_squared_hidden_words[i] for i in indexs]
            t +=1
        return best_k_seqs, best_k_probs, best_k_hidden_edits,best_k_hidden_words,best_k_counters


class EditNTS(nn.Module):
    def __init__(self, config, n_layers=2):
        super(EditNTS, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if not(config.pretrained_embedding is None):
            print('load pre-trained embeddings')
            self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.embeddingPOS = nn.Embedding(config.pos_vocab_size, config.pos_embedding_dim)

        self.encoder1 = EncoderRNN(config.vocab_size, config.embedding_dim,
                                   config.pos_vocab_size, config.pos_embedding_dim,
                                   config.word_hidden_units,
                                   n_layers,
                                   self.embedding, self.embeddingPOS)

        self.decoder = EditDecoderRNN(config.vocab_size, config.embedding_dim, config.word_hidden_units * 2,
                                      n_layers, self.embedding)


    def forward(self,org,output,org_ids,org_pos,simp_sent,teacher_forcing_ratio=1.0):
        def transform_hidden(hidden): #for bidirectional encoders
            h, c = hidden
            h = torch.cat([h[0], h[1]], dim=1)[None, :, :]
            c = torch.cat([c[0], c[1]], dim=1)[None, :, :]
            hidden = (h, c)
            return hidden
        hidden_org = self.encoder1.initHidden(org[0].size(0))
        encoder_outputs_org, hidden_org = self.encoder1(org,org_pos,hidden_org)
        hidden_org = transform_hidden(hidden_org)

        logp, _ = self.decoder(output, hidden_org, encoder_outputs_org,org_ids,simp_sent,teacher_forcing_ratio)
        return logp

    def beamsearch(self, org, input_edits,simp_sent,org_ids,org_pos, beam_size=5):
        def transform_hidden(hidden): #for bidirectional encoders
            h, c = hidden
            h = torch.cat([h[0], h[1]], dim=1)[None, :, :]
            c = torch.cat([c[0], c[1]], dim=1)[None, :, :]
            hidden = (h, c)
            return hidden
        hidden_org = self.encoder1.initHidden(org[0].size(0))
        encoder_outputs_org, hidden_org = self.encoder1(org,org_pos,hidden_org)
        hidden_org = transform_hidden(hidden_org)


        best_k_seqs, best_k_probs, best_k_hidden_edits, best_k_hidden_words, best_k_counters =\
            self.decoder.beam_forward(input_edits,simp_sent, hidden_org, encoder_outputs_org,org_ids,beam_size)

        best_seq_list=[]
        for sq in best_k_seqs:
            best_seq_list.append([i.item() for i in sq])

        # find final best output
        index = np.argsort(best_k_probs)[::-1][0]
        best_seq = best_k_seqs[index]
        best_seq_np=[i.item() for i in best_seq]
        return best_seq_list



