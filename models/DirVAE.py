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
        prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        prior_var = torch.from_numpy((((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +
                                 (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        prior_logvar = prior_var.log()

        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

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
        # sentene encoder
        if self.sent_type == "bow":
            input_embedding_size = output_embedding_size = self.embed_size
        elif self.sent_type == "rnn":
            self.sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, self.keep_prob, 1)
            input_embedding_size = output_embedding_size = self.sent_cell_size
        elif self.sent_type == "bi_rnn":
            self.bi_sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, keep_prob=1.0, num_layer=1, bidirectional=True)
            input_embedding_size = output_embedding_size = self.sent_cell_size * 2

        joint_embedding_size = input_embedding_size + 2

        # contextRNN for input context and profile
        self.enc_cell = self.get_rnncell(config.cell_type, joint_embedding_size, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)
        cond_embedding_size = self.context_cell_size * 2 if config.use_profile else self.context_cell_size

        # recoginitionNetwork for response, with approximated Dirchlet function
        recog_input_size = output_embedding_size
        self.logvar_fc = nn.Linear(recog_input_size, self.h_dim)
        self.mean_fc = nn.Linear(recog_input_size, self.h_dim)
        self.mean_bn    = nn.BatchNorm1d(self.h_dim)                   # bn for mean
        self.logvar_bn  = nn.BatchNorm1d(self.h_dim)               # bn for logvar
        self.logvar_bn.weight.requires_grad = False
        self.mean_bn.weight.requires_grad = False
        
        # generation work in topicVae logp(x|z)p(z)
        self.dec_init_state_net = nn.Linear(self.h_dim, self.dec_cell_size)
        dec_input_embedding_size = self.embed_size
        self.rec_dec_cell = self.get_rnncell(config.cell_type, dec_input_embedding_size, self.dec_cell_size, config.keep_prob, config.num_layer)
        self.dec_cell_proj = nn.Linear(self.dec_cell_size, self.vocab_size)

        # generation work with latent and condition
        dec_all_input_size = self.h_dim + cond_embedding_size
        self.dec_init_state_net_all = nn.Linear(dec_all_input_size, self.dec_cell_size)
        self.dec_cell = self.get_rnncell(config.cell_type, dec_input_embedding_size, self.dec_cell_size, config.keep_prob, config.num_layer)
        self.dec_cell_proj_all = nn.Linear(self.dec_cell_size, self.vocab_size)

        # BOW loss
        self.bow_project = nn.Sequential(
            nn.Linear(self.h_dim, 400),
            nn.Tanh(),
            nn.Dropout(1 - config.keep_prob),
            nn.Linear(400, self.vocab_size)
        )

        self.build_optimizer(config, log_dir)

        # initilize learning rate
        self.learning_rate = config.init_lr
        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.input_contexts = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="dialog_context")
            self.floors = tf.placeholder(dtype=tf.int32, shape=(None, None), name="floor")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")
            self.topics = tf.placeholder(dtype=tf.int32, shape=(None,), name="topics")
            #self.my_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="my_profile")
            #self.ot_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="ot_profile")

            # target response given the dialog context
            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")
            self.output_das = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_dialog_acts")

            # optimization related variables
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

    def learning_rate_decay():
        self.learning_rate = self.learning_rate * config.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def forward(self, feed_dict, mode='train', use_profile=False):
        for k, v in feed_dict.items():
            setattr(self, k, v)

        max_dialog_len = self.input_contexts.size(1)
        if use_profile:
            max_profile_len = self.profile_contexts.size(1)

        with variable_scope.variable_scope("wordEmbedding"):

            self.input_contexts = self.input_contexts.view(-1, self.max_utt_len)
            input_embedding = self.embedding(self.input_contexts)
            output_embedding = self.embedding(self.output_tokens)
            if use_profile:
                self.profile_contexts = self.profile_contexts.view(-1, self.max_utt_len)
                profile_embedding = self.embedding(self.profile_contexts)

            assert ((self.input_contexts.view(-1, self.max_utt_len) > 0).float() - (torch.max(torch.abs(input_embedding), 2)[0] > 0).float()).abs().sum().item() == 0,\
                str(((self.input_contexts.view(-1, self.max_utt_len) > 0).float() - (torch.max(torch.abs(input_embedding), 2)[0] > 0).float()).abs().sum().item())

            if self.sent_type == "bow":
                input_embedding, sent_size = get_bow(input_embedding)
                output_embedding, _ = get_bow(output_embedding)
                if use_profile:
                    profile_embedding, p_sent_size = get_bow(profile_embedding)

            elif self.sent_type == "rnn":
                input_embedding, sent_size = get_rnn_encode(input_embedding, self.sent_cell, self.keep_prob, scope="sent_rnn")
                output_embedding, _ = get_rnn_encode(output_embedding, self.sent_cell, self.output_lens,
                                                     self.keep_prob, scope="sent_rnn", reuse=True)
                if use_profile:
                    profile_embedding, p_sent_size = get_rnn_encode(profile_embedding, self.sent_cell, self.keep_prob, scope="sent_rnn")
            elif self.sent_type == "bi_rnn":
                input_embedding, sent_size = get_bi_rnn_encode(input_embedding, self.bi_sent_cell, scope="sent_bi_rnn")
                output_embedding, _ = get_bi_rnn_encode(output_embedding, self.bi_sent_cell, self.output_lens, scope="sent_bi_rnn", reuse=True)
                if use_profile:
                    profile_embedding, p_sent_size = get_bi_rnn_encode(profile_embedding, self.bi_sent_cell, scope="sent_bi_rnn")
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs
            input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)
            if use_profile:
                profile_embedding = profile_embedding.view(-1, max_profile_len, p_sent_size)
            if self.keep_prob < 1.0:
                input_embedding = F.dropout(input_embedding, 1 - self.keep_prob, self.training)
                if use_profile:
                    profile_embedding = F.dropout(profile_embedding, 1 - self.keep_prob, self.training)

            # convert floors into 1 hot
            floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
            floor_one_hot.data.scatter_(1, self.floors.view(-1,1), 1)
            floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)
            joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)
            if use_profile:
                profile_post = torch.zeros(floor_one_hot.size()[0], max_profile_len, 2).cuda()
                joint_embedding_profile = torch.cat([profile_embedding, profile_post], 2)

        with variable_scope.variable_scope("contextRNN"):
            # and enc_last_state will be same as the true last state
            # self.enc_cell.eval()
            _, enc_last_state = utils.dynamic_rnn(
                self.enc_cell,
                joint_embedding,
                sequence_length=self.context_lens)

            if use_profile:
                _, enc_last_state_profile = utils.dynamic_rnn(
                    self.enc_cell,
                    joint_embedding_profile,
                    sequence_length=self.profile_lens)

            if self.num_layer > 1:
                enc_last_state = torch.cat([_ for _ in torch.unbind(enc_last_state)], 1)
                if use_profile:
                    enc_last_state_profile = torch.cat([_ for _ in torch.unbind(enc_last_state_profile)], 1)
            else:
                enc_last_state = enc_last_state.squeeze(0)
                if use_profile:
                    enc_last_state_profile = enc_last_state_profile.squeeze(0)

        cond_embedding = torch.cat([enc_last_state, enc_last_state_profile], 1) if use_profile else enc_last_state

        with variable_scope.variable_scope("recognitionNetwork"):
            recog_input = output_embedding
            posterior_mean   = self.mean_bn  (self.mean_fc  (recog_input))          # posterior mean
            posterior_logvar = self.logvar_bn(self.logvar_fc(recog_input))          # posterior log variance
            posterior_var    = posterior_logvar.exp()

            # take sample
            eps = posterior_mean.data.new().resize_as_(posterior_mean.data).normal_(0,1) # noise
            z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
            self.p = F.softmax(z, -1)    
            if self.keep_prob < 1.0:
                self.p = F.dropout(self.p, 1 - self.keep_prob, self.training)


        with variable_scope.variable_scope("recognitionGeneration"):
            dec_init = self.dec_init_state_net(self.p).unsqueeze(0) 

            # BOW loss
            self.bow_logits = self.bow_project(self.p)
        
        with variable_scope.variable_scope("recogDecoder"):
            if mode == 'test':
                pass
            else:
                input_tokens = self.output_tokens[:, :-1]
                if self.dec_keep_prob < 1.0:
                        # if token is 0, then embedding is 0, it's the same as word drop
                        keep_mask = input_tokens.new_empty(input_tokens.size()).bernoulli_(config.dec_keep_prob)
                        input_tokens = input_tokens * keep_mask
                dec_input_embedding = self.embedding(input_tokens)
                dec_seq_lens = self.output_lens - 1
                dec_input_embedding = F.dropout(dec_input_embedding, 1 - self.keep_prob, self.training)
                dec_outs, _, final_context_state =  decoder_fn_lib.train_loop(self.dec_cell, 
                                                                              self.dec_cell_proj, 
                                                                              dec_input_embedding,
                                                                              init_state=dec_init, 
                                                                              context_vector=None, 
                                                                              sequence_length=dec_seq_lens)

                if final_context_state is not None:
                    self.dec_out_words = final_context_state
                else:
                    self.dec_out_words = torch.max(dec_outs, 2)[1]
        
        if not mode == 'test':
            with variable_scope.variable_scope("loss"):
                labels = self.output_tokens[:, 1:]
                label_mask = torch.sign(labels).detach().float()

                # rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)
                rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)), labels.reshape(-1), reduce=False).view(dec_outs.size()[:-1])
                # print(rc_loss * label_mask)
                rc_loss = torch.sum(rc_loss * label_mask, 1)
                self.avg_rc_loss = rc_loss.mean()
                # used only for perpliexty calculation. Not used for optimzation
                self.rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

                """ as n-trial multimodal distribution. """
                # bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * label_mask
                bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
                bow_loss = torch.sum(bow_loss, 1)
                self.avg_bow_loss  = torch.mean(bow_loss)

                prior_mean   = self.prior_mean.expand_as(posterior_mean)
                prior_var    = self.prior_var.expand_as(posterior_mean)
                prior_logvar = self.prior_logvar.expand_as(posterior_mean)
                var_division    = posterior_var  / prior_var
                diff            = posterior_mean - prior_mean
                diff_term       = diff * diff / prior_var
                logvar_division = prior_logvar - posterior_logvar
                # put KLD together
                KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.h_dim )
                self.avg_kld = torch.mean(KLD)
                if mode == 'train':
                    kl_weights = min(self.global_t / self.full_kl_step, 1.0)
                else:
                    kl_weights = 1.0
                
                self.kl_w = kl_weights
                self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld
                self.aug_elbo = self.avg_bow_loss + self.elbo

                self.summary_op = [\
                    tb.summary.scalar("model/loss/rc_loss", self.avg_rc_loss.item()),
                    tb.summary.scalar("model/loss/elbo", self.elbo.item()),
                    tb.summary.scalar("model/loss/kld", self.avg_kld.item()),
                    tb.summary.scalar("model/loss/bow_loss", self.avg_bow_loss.item())]
         
    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):
        context, context_lens, floors, topics, my_profiles, ot_profiles, outputs, output_lens, output_das, p_context, p_lens = batch
        feed_dict = {"input_contexts": context, "context_lens":context_lens,
                     "floors": floors, "topics":topics, "my_profile": my_profiles,
                     "ot_profile": ot_profiles, "output_tokens": outputs,
                     "output_das": output_das, "output_lens": output_lens,
                     "use_prior": use_prior, "profile_contexts": p_context, "profile_lens":p_lens}
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key == "use_prior":
                    tiled_feed_dict[key] = val
                    continue
                if val is None:
                    tiled_feed_dict[key] = None
                    continue
                multipliers = [1]*len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict["global_t"] = global_t

        if torch.cuda.is_available():
            feed_dict = {k: torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v for k, v in feed_dict.items()}
        else:
            feed_dict = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in feed_dict.items()}

        return feed_dict

    def train_model(self, global_t, train_feed, update_limit=5000, use_profile=False):
            elbo_losses = []
            rc_losses = []
            rc_ppls = []
            kl_losses = []
            bow_losses = []
            local_t = 0
            start_time = time.time()
            loss_names =  ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"]
            while True:
                batch = train_feed.next_batch()
                if batch is None:
                    break
                if update_limit is not None and local_t >= update_limit:
                    break
                feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
                self.forward(feed_dict, mode='train', use_profile=use_profile)
                elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = self.elbo.item(),\
                                                                self.avg_bow_loss.item(),\
                                                                self.avg_rc_loss.item(),\
                                                                self.rc_ppl.item(),\
                                                                self.avg_kld.item()

                self.optimize(self.aug_elbo)
                # print(elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss)
                for summary in self.summary_op:
                    self.train_summary_writer.add_summary(summary, global_t)
                elbo_losses.append(elbo_loss)
                bow_losses.append(bow_loss)
                rc_ppls.append(rc_ppl)
                rc_losses.append(rc_loss)
                kl_losses.append(kl_loss)

                global_t += 1
                local_t += 1
                if local_t % (train_feed.num_batch // 10) == 0:
                    kl_w = self.kl_w
                    self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                    loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "kl_w %f" % kl_w)

            # finish epoch!
            torch.cuda.synchronize()
            epoch_time = time.time() - start_time
            avg_losses = self.print_loss("Epoch Done", loss_names,
                                        [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                        "step time %.4f" % (epoch_time / train_feed.num_batch))

            return global_t, avg_losses[0]

    def valid_model(self, name, valid_feed, use_profile=False):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)
            with torch.no_grad():
                self.forward(feed_dict, mode='valid',  use_profile=use_profile)
            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = self.elbo.item(),\
                                                            self.avg_bow_loss.item(),\
                                                            self.avg_rc_loss.item(),\
                                                            self.rc_ppl.item(),\
                                                            self.avg_kld.item()
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            bow_losses.append(bow_loss)
            kl_losses.append(kl_loss)

        avg_losses = self.print_loss(name, ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"],
                                    [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "")
        return avg_losses, ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"]
