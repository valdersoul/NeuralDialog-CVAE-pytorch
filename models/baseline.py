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

class S2Smemory(BaseTFModel):
    '''
    Sequence-to-sequence baseline with attention for persona generation dataset, with/without
    pfofiles.
    When using profiles, we will use attention memory together with the input
    '''
    def __init__(self, config, api, log_dir, scope=None):
        super(S2Smemory, self).__init__()
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

        # self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
        #         # self.idxembedding = nn.Embedding(self.vocab_size, 1)
        #         #
        #         # self.embedding.weight.require_grad = False
        #         # self.idxembedding.weight.require_grad = False
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.array(api.word2vec, dtype='float32')))
        self.idxembedding = nn.Embedding.from_pretrained(torch.from_numpy(np.array(api.word2idx, dtype='float32')).unsqueeze(1))

        # no dropout at last layer, we need to add one
        if self.sent_type == "bow":
            input_embedding_size = output_embedding_size = self.embed_size
        elif self.sent_type == "rnn":
            self.sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, self.keep_prob, 1)
            input_embedding_size = output_embedding_size = self.sent_cell_size
        elif self.sent_type == "bi_rnn":
            self.bi_sent_cell = self.get_rnncell("gru", self.embed_size, self.sent_cell_size, keep_prob=1.0, num_layer=1, bidirectional=True)
            input_embedding_size = output_embedding_size = self.sent_cell_size * 2

        # embedding + 1/0 identify encoding
        joint_embedding_size = input_embedding_size + 2

        self.enc_cell = self.get_rnncell(config.cell_type, joint_embedding_size, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)
        cond_embedding_size = config.topic_embed_size + self.context_cell_size

        dec_inputs_size = cond_embedding_size

        # Decoder
        if config.num_layer > 1:
            self.dec_init_state_net = nn.ModuleList(
                [nn.Linear(dec_inputs_size, self.dec_cell_size) for i in range(config.num_layer)])
        else:
            self.dec_init_state_net = nn.Sequential(nn.Linear(dec_inputs_size, self.dec_cell_size), nn.Tanh())

        # decoder
        dec_input_embedding_size = self.embed_size if not self.use_profile else self.embed_size + input_embedding_size

        self.dec_cell = self.get_rnncell(config.cell_type, dec_input_embedding_size, self.dec_cell_size,
                                         config.keep_prob, config.num_layer)
        self.dec_cell_proj = nn.Linear(self.dec_cell_size, self.vocab_size)
        self.atten_proj = nn.Linear(input_embedding_size, self.dec_cell_size)

        self.build_optimizer(config, log_dir)

        # initilize learning rate
        self.learning_rate = config.init_lr
        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.input_contexts = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="dialog_context")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")
            self.profile_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="profile_lens")
            #self.my_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="my_profile")
            #self.ot_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="ot_profile")

            # target response given the dialog context
            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")

            # optimization related variables
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

    def forward(self, feed_dict, use_profile=False, mode='train'):
        for k, v in feed_dict.items():
            setattr(self, k, v)

        max_dialog_len = self.input_contexts.size(1)
        if use_profile:
            max_profile_len = self.profile_contexts.size(1)

        with variable_scope.variable_scope("wordEmbedding"):
            self.input_contexts = self.input_contexts.view(-1, self.max_utt_len)
            input_embedding = self.embedding(self.input_contexts)
            if use_profile:
                profile_mask = (self.profile_contexts.sum(-1) != 0).float()
                self.profile_contexts = self.profile_contexts.view(-1, self.max_utt_len)
                profile_embedding = self.embedding(self.profile_contexts)

            if self.sent_type == "bow":
                input_embedding, sent_size = get_bow(input_embedding)
                if use_profile:
                    profile_embedding, p_sent_size = get_bow(profile_embedding)

            elif self.sent_type == "rnn":
                input_embedding, sent_size = get_rnn_encode(input_embedding, self.sent_cell, self.keep_prob,
                                                            scope="sent_rnn")
                if use_profile:
                    profile_embedding, p_sent_size = get_rnn_encode(profile_embedding, self.sent_cell,
                                                                    self.keep_prob, scope="sent_rnn")
            elif self.sent_type == "bi_rnn":
                input_embedding, sent_size = get_bi_rnn_encode(input_embedding, self.bi_sent_cell,
                                                               scope="sent_bi_rnn")
                if use_profile:
                    profile_embedding, p_sent_size = get_bi_rnn_encode(profile_embedding, self.bi_sent_cell,
                                                                       scope="sent_bi_rnn")
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs
            input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)
            if use_profile:
                profile_embedding = profile_embedding.view(-1, max_profile_len, p_sent_size)
            if self.keep_prob < 1.0:
                input_embedding = F.dropout(input_embedding, 1 - self.keep_prob, self.training)

            # convert floors into 1 hot
            floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
            floor_one_hot.data.scatter_(1, self.floors.view(-1, 1), 1)
            floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)
            joint_embedding_input = torch.cat([input_embedding, floor_one_hot], 2)

            # self.input_contexts = self.input_contexts.view(-1, self.max_utt_len)
            # input_embedding = self.embedding(self.input_contexts)
            # if self.sent_type == "bow":
            #     input_embedding, sent_size = get_bow(input_embedding)
            # elif self.sent_type == "rnn":
            #     input_embedding, sent_size = get_rnn_encode(input_embedding, self.sent_cell, self.keep_prob, scope="sent_rnn")
            # elif self.sent_type == "bi_rnn":
            #     input_embedding, sent_size = get_bi_rnn_encode(input_embedding, self.bi_sent_cell, scope="sent_bi_rnn")
            # else:
            #     raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")
            # # reshape input into dialogs
            # input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)
            # if use_profile:
            #     self.profile_contexts = self.profile_contexts.view(-1, self.max_utt_len)
            #     profile_embedding = self.embedding(self.profile_contexts)
            #     profile_idx = self.idxembedding(self.profile_contexts)
            #     profile_embedding, p_sent_size = get_idf(profile_embedding, profile_idx)
            #     profile_embedding = profile_embedding.view(-1, max_profile_len, p_sent_size)
            # if self.keep_prob < 1.0:
            #     input_embedding = F.dropout(input_embedding, 1 - self.keep_prob, self.training)
            #
            # # convert floors into 1 hot
            # floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
            # floor_one_hot.data.scatter_(1, self.floors.view(-1,1), 1)
            # floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)
            # joint_embedding_input = torch.cat([input_embedding, floor_one_hot], 2)

        with variable_scope.variable_scope("contextRNN"):
            # and enc_last_state will be same as the true last state
            # self.enc_cell.eval()
            _, enc_last_state = utils.dynamic_rnn(
                self.enc_cell,
                joint_embedding_input,
                sequence_length=self.context_lens)

            if self.num_layer > 1:
                enc_last_state = torch.cat([_ for _ in torch.unbind(enc_last_state)], 1)
            else:
                enc_last_state = enc_last_state.squeeze(0)

        with variable_scope.variable_scope("generationNetwork"):
            dec_inputs = enc_last_state

            # Decoder
            if self.num_layer > 1:
                dec_init_state = [self.dec_init_state_net[i](dec_inputs) for i in range(self.num_layer)]
                dec_init_state = torch.stack(dec_init_state)
            else:
                dec_init_state = self.dec_init_state_net(dec_inputs).unsqueeze(0)

        with variable_scope.variable_scope("decoder"):
            if mode == 'test':
                dec_outs, _, final_context_state = decoder_fn_lib.inference_loop(self.dec_cell, self.dec_cell_proj, self.embedding,
                                                                    encoder_state = dec_init_state,
                                                                    start_of_sequence_id=self.go_id,
                                                                    end_of_sequence_id=self.eos_id,
                                                                    maximum_length=self.max_utt_len,
                                                                    num_decoder_symbols=self.vocab_size,
                                                                    context_vector=None,
                                                                    decode_type='greedy')
            else:
                # loop_func = decoder_fn_lib.context_decoder_fn_train(dec_init_state, selected_attribute_embedding)
                # apply word dropping. Set dropped word to 0
                input_tokens = self.output_tokens[:, :-1]
                if self.dec_keep_prob < 1.0:
                    # if token is 0, then embedding is 0, it's the same as word drop
                    keep_mask = input_tokens.new_empty(input_tokens.size()).bernoulli_(config.dec_keep_prob)
                    input_tokens = input_tokens * keep_mask

                dec_input_embedding = self.embedding(input_tokens)
                dec_seq_lens = self.output_lens - 1

                # Apply embedding dropout
                dec_input_embedding = F.dropout(dec_input_embedding, 1 - self.keep_prob, self.training)

                dec_outs, _, final_context_state =  decoder_fn_lib.train_attention_loop(self.dec_cell, 
                                                                                        self.dec_cell_proj, 
                                                                                        dec_input_embedding,
                                                                                        atten_fn=self.atten_proj,
                                                                                        init_state=dec_init_state, 
                                                                                        context_vector=profile_embedding, 
                                                                                        max_length=max_dialog_len,
                                                                                        atten_mask=profile_mask)

            if final_context_state is not None:
                self.dec_out_words = final_context_state
            else:
                self.dec_out_words = torch.max(dec_outs, 2)[1]

        if not mode == 'test':
            with variable_scope.variable_scope("loss"):
                labels = self.output_tokens[:, 1:]
                label_mask = torch.sign(labels).detach().float()

                # rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)
                rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)), labels.reshape(-1), reduction='none').view(
                    dec_outs.size()[:-1])
                # print(rc_loss * label_mask)
                rc_loss = torch.sum(rc_loss * label_mask, 1)
                self.avg_rc_loss = rc_loss.mean()
                # used only for perpliexty calculation. Not used for optimzation
                self.rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

                self.summary_op = [ \
                    tb.summary.scalar("model/loss/rc_loss", self.avg_rc_loss.item())]

                # self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
                # self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
                # self.est_marginal = torch.mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

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
        rc_losses = []
        rc_ppls = []
        local_t = 0
        start_time = time.time()
        loss_names =  ["rc_loss", "rc_peplexity"]
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
            self.forward(feed_dict, use_profile=use_profile, mode='train')
            rc_loss, rc_ppl = self.avg_rc_loss.item(), self.rc_ppl.item()

            self.optimize(self.avg_rc_loss)
            # print(elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss)
            for summary in self.summary_op:
                self.train_summary_writer.add_summary(summary, global_t)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch // 10) == 0:
                kl_w = 0
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [rc_losses, rc_ppls], "kl_w %f" % kl_w)

        # finish epoch!
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [rc_losses, rc_ppls],
                                     "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def valid_model(self, name, valid_feed, use_profile=False):
        rc_losses = []
        rc_ppls = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)
            with torch.no_grad():
                self.forward(feed_dict, use_profile=use_profile, mode='valid')
            rc_loss, rc_ppl = self.avg_rc_loss.item(), self.rc_ppl.item()
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)


        avg_losses = self.print_loss(name, ["rc_loss", "rc_peplexity"],
                                     [rc_losses, rc_ppls], "")
        return avg_losses, ["rc_loss", "rc_peplexity"]

    def test_model(self, test_feed, num_batch=None, repeat=5, dest=sys.stdout, use_profile=False):
        local_t = 0
        recall_bleus = []
        prec_bleus = []

        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=1)
            with torch.no_grad():
                self.forward(feed_dict, mode='test', use_profile=use_profile)
            word_outs = self.dec_out_words.cpu().numpy()
            sample_words = word_outs #np.split(word_outs, repeat, axis=0)

            true_floor = feed_dict["floors"].cpu().numpy()
            true_srcs = feed_dict["input_contexts"].cpu().numpy()
            true_src_lens = feed_dict["context_lens"].cpu().numpy()
            true_outs = feed_dict["output_tokens"].cpu().numpy()
            profile = feed_dict["profile_contexts"].cpu().numpy()
            #true_topics = feed_dict["topics"].cpu().numpy()
            #true_das = feed_dict["output_das"].cpu().numpy()
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch // 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

            for b_id in range(test_feed.batch_size):
                # print the dialog context
                start = np.maximum(0, true_src_lens[b_id]-5)
                for t_id in range(start, true_srcs.shape[1], 1):
                    src_str = " ".join([self.vocab[e] for e in true_srcs[b_id, t_id].tolist() if e != 0])
                    dest.write("Src %d-%d: %s\n" % (t_id, true_floor[b_id, t_id], src_str))
                for p_id in range(profile.shape[1]):
                    profile_str = " ".join([self.vocab[e] for e in profile[b_id, p_id].tolist() if e != 0])
                    dest.write("Profile %d-%d: %s\n" % (p_id, 1, profile_str))
                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                #da_str = self.da_vocab[true_das[b_id]]
                # print the predicted outputs
                dest.write("Target  >> %s\n" % ( true_str))
                local_tokens = []

                pred_outs = sample_words
                #pred_da = np.argmax(sample_das[r_id], axis=1)[0]
                pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                dest.write("Sample %d  >> %s\n" % (0, pred_str))
                local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = utils.get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                # make a new line for better readability
                dest.write("\n")

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")
        print("Done testing")
