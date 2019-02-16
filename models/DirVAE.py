from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
import sys
from tensorflow.python.ops import variable_scope

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import decoder_fn_lib
import numpy as np
import re
from . import utils
from .utils import sample_gaussian, gaussian_kld, norm_log_liklihood, get_bow, get_rnn_encode, get_bi_rnn_encode, get_idf

import tensorboardX as tb
import tensorboardX.summary
import tensorboardX.writer

from .cvae import BaseTFModel

class DirVAE(BaseTFModel):
    '''
    Sequence-to-sequence baseline with attention for persona generation dataset, with/without
    pfofiles.
    When using profiles, we will use attention memory together with the input
    '''
    def __init__(self, config, api, log_dir, scope=None):
        super(DirVAE, self).__init__()

        # The approximated Dirchlet function prior
        self.h_dim = config.num_topic
        self.a = 1.*np.ones((1 , self.h_dim)).astype(np.float32)
        self.prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        self.prior_var = torch.from_numpy((((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +
                                 (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        self.prior_logvar = prior_var.log()

        self.use_profile = config.use_profile
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)

        self.scope = scope
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size

        self.embed_size = config.embed_size
        self.sent_type = config.sent_type
        self.keep_prob = config.keep_prob
        self.num_layer = config.num_layer
        self.dec_keep_prob = config.dec_keep_prob
        self.full_kl_step = config.full_kl_step
        self.grad_clip = config.grad_clip
        self.grad_noise = config.grad_noise

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.array(api.word2vec, dtype='float32')))

        # no dropout at last layer, we need to add one
        if self.sent_type == "bow":
            input_embedding_size = output_embedding_size = self.embed_size
        elif self.sent_type == "rnn":
            self.sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, self.keep_prob, 1)
            input_embedding_size = output_embedding_size = self.sent_cell_size
        elif self.sent_type == "bi_rnn":
            self.bi_sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, keep_prob=1.0, num_layer=1, bidirectional=True)
            input_embedding_size = output_embedding_size = self.sent_cell_size * 2

        joint_embedding_size = input_embedding_size + 2
