#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def get_bleu_stats(ref, hyps):
    scores = []
    for hyp in hyps:
        try:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3,1./3]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5*torch.sum(logvar + np.log(2*np.pi) + torch.div(torch.pow((x-mu), 2), torch.exp(logvar)), 1)


def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z= mu + std * epsilon
    return z


def get_idf(embedding, mask):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    embedding_size = embedding.size(2)
    tf = 1e6 / (torch.pow(mask, 1.07))
    weights = 1 / (1 + torch.log(1 + tf))
    embedding = embedding * weights
    return embedding.sum(1), embedding_size

def get_bow(embedding, avg=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    embedding_size = embedding.size(2)
    if avg:
        return embedding.mean(1), embedding_size
    else:
        return embedding.sum(1), embedding_size

def dynamic_rnn(cell, inputs, sequence_length, init_state=None, output_fn=None):
    sorted_lens, len_ix = sequence_length.sort(0, descending=True)

    # Used for later reorder
    inv_ix = len_ix.clone()
    inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

    # The number of inputs that have lengths > 0
    valid_num = torch.sign(sorted_lens).long().sum().item()
    zero_num = inputs.size(0) - valid_num
    # print('zero_num:', zero_num)

    sorted_inputs = inputs[len_ix].contiguous()
    if init_state is not None:
        sorted_init_state = init_state[:, len_ix].contiguous()
    
    packed_inputs = pack_padded_sequence(sorted_inputs[:valid_num], list(sorted_lens[:valid_num]), batch_first=True)

    if init_state is not None:
        outputs, state = cell(packed_inputs, sorted_init_state[:, :valid_num])
    else:
        outputs, state = cell(packed_inputs)

    # Reshape *final* output to (batch_size, hidden_size)
    outputs, _ = pad_packed_sequence(outputs, batch_first=True)

    # Add back the zero lengths
    if zero_num > 0:
        outputs = torch.cat([outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)
        if init_state is not None:
            state = torch.cat([state, sorted_init_state[:, valid_num:]], 1)
        else:
            state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

    # Reorder to the original order
    outputs = outputs[inv_ix].contiguous()
    state = state[:, inv_ix].contiguous()

    # compensate the last last layer dropout, necessary????????? need to check!!!!!!!!
    state = F.dropout(state, cell.dropout, cell.training)
    outputs = F.dropout(outputs, cell.dropout, cell.training)

    if output_fn is not None:
        outputs = output_fn(outputs)

    return outputs, state

def attention(hidden, context_vector, atten_fn=None, atten_mask=None):
    """
    Compute the profile-aware attentioned context vector
    """
    if atten_fn is not None:
        projected_context = F.tanh(atten_fn(context_vector))
    else:
        projected_context = context_vector
    hidden = hidden.permute(1, 2, 0)
    weights = torch.bmm(projected_context, hidden).squeeze()
    if atten_mask is not None:
        mask = atten_mask
    else:
        mask = (weights != 0).float()
    weights = torch.exp(weights - weights.max(-1, keepdim=True)[0]) * mask
    weights = weights / weights.sum(1, keepdim=True)
    context = (context_vector * weights.unsqueeze(2)).sum(1)

    return context

def decode_once(cell, input, hidden_state, context_vector, atten_fn=None, atten_mask=None):
    """
    compute the attentioned input and output the result and the next hidden
    """
    context = attention(hidden_state, context_vector, atten_fn, atten_mask)
    enhance_input = torch.cat([context, input], -1)
    output, hidden_state = cell(enhance_input.unsqueeze(1), hidden_state)
    return output, hidden_state

def dynamic_rnn_attention(cell, inputs, max_length, init_state, output_fn, atten_fn, context_vector, atten_mask=None):
    """
    Used to decode with attention
    """
    hidden_state = init_state
    outputs = []
    length = inputs.size(1)
    for time in range(length):
        input = inputs[:,time,:]
        output, hidden_state = decode_once(cell, input, hidden_state, context_vector, atten_fn, atten_mask)
        outputs.append(output)

    outputs = torch.cat(outputs, 1)
    # compensate the last last layer dropout, necessary????????? need to check!!!!!!!!
    state = F.dropout(hidden_state, cell.dropout, cell.training)
    outputs = F.dropout(outputs, cell.dropout, cell.training)

    if output_fn is not None:
        outputs = output_fn(outputs)

    return outputs, state
    


def get_rnn_encode(embedding, cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
            length_mask = length_mask.long()
        _, encoded_input = dynamic_rnn(cell, embedding, sequence_length=length_mask)

        # get only the last layer
        encoded_input = encoded_input[-1]
        return encoded_input, rnn.hidden_size


def get_bi_rnn_encode(embedding, cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
            length_mask = length_mask.long()
        _, encoded_input = dynamic_rnn(cell, embedding, sequence_length=length_mask)
        # get only the last layer
        encoded_input = torch.cat([encoded_input[-2], encoded_input[-1]], 1)
        return encoded_input, cell.hidden_size * 2
