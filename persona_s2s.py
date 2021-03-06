import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import ProfileBaselineConfig as Config
from data_apis.corpus import SWDADialogCorpus, PERSONADialogCorpus
from data_apis.data_utils import PERSONAataLoader
from data_apis.data_utils import SWDADataLoader
from models.cvae import KgRnnCVAE, S2S
from models.baseline import S2Smemory
from models.DirVAE import DirVAE
import torch

import glob

# constants
tf.app.flags.DEFINE_string("word2vec_path", "embeddings/glove_twitter_27B_200d.txt", "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data/full_swda_clean_42da_sentiment_dialog_corpus.p", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "run1500783422", "the dir to load checkpoint for forward only")
tf.app.flags.DEFINE_string("model", "cvae", "model used to train/valid")
tf.app.flags.DEFINE_string("data", "both", "data used to train/valid")
tf.app.flags.DEFINE_integer("number", 0, 'model to load')
FLAGS = tf.app.flags.FLAGS

def get_checkpoint_state(ckp_dir):
    files = os.path.join(ckp_dir, "*.pth")
    files = glob.glob(files)
    files.sort(key=os.path.getmtime)
    return len(files) > 0 and files[-1] or None

def main():
    # config for training
    config = Config()
    if FLAGS.data == 'none':
        config.use_profile = False

    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 60

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1

    train_persona = None
    valid_persona = None

    pp(config)
    data_path = "data/convai2/train_" + FLAGS.data + "_original_no_cands.txt"
    # get data set
    api = PERSONADialogCorpus(data_path, FLAGS.data, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)
    print("dataset loaded")

    dial_corpus = api.get_dialog_corpus()
    train_dial, valid_dial = dial_corpus.get("train"), dial_corpus.get("valid")
    if config.use_profile:
        persona_corpus = api.get_persona_corpus()
        train_persona, valid_persona = persona_corpus.get("train"), persona_corpus.get("valid")

    # convert to numeric input outputs that fits into TF models
    train_feed = PERSONAataLoader("Train", train_dial, train_persona, config)
    valid_feed = PERSONAataLoader("Valid", valid_dial, valid_persona, config)

    if FLAGS.forward_only or FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.model)

    # begin training
    if True:
        scope = "model"
        if FLAGS.model == 'cvae':
            model = DirVAE(config, api, log_dir=None if FLAGS.forward_only else log_dir, scope=scope)
        elif FLAGS.model == 's2s':
            model = S2S(config, api, log_dir=None if FLAGS.forward_only else log_dir, scope=scope)

        print("Created computation graphs")
        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "run.log"), "wb") as f:
                f.write(pp(config, output=False))

        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        model.apply(lambda m: [torch.nn.init.uniform_(p.data, -1.0 * config.init_w, config.init_w) for p in m.parameters()])

        # Load word2vec weight
        if api.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            model.embedding.weight.data.copy_(torch.from_numpy(np.array(api.word2vec)))
            model.embedding.weight.require_grad = False
        model.embedding.weight.data[0].fill_(0)
        #model.idxembedding.weight.data.copy_(torch.from_numpy(np.array(api.word2idx, dtype='float32')).unsqueeze(-1))
        #model.idxembedding.weight.require_grad = False

        if ckpt:
            print("Reading dm models parameters from %s" % ckpt)
            model.load_state_dict(torch.load(ckpt))

        if torch.cuda.is_available():
            model.cuda()

        if not FLAGS.forward_only:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ "-%d.pth")
            global_t = 1
            patience = 10  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            best_dev_losses = None
            loss_names = None
            for epoch in range(config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate))

                # begin training
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=True)
                global_t, train_loss = model.train_model(global_t, train_feed, update_limit=config.update_limit, use_profile=config.use_profile)

                # begin validation
                # valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                #                       valid_config.step_size, shuffle=False, intra_shuffle=False)
                # model.eval()
                # model.test_model(valid_feed, num_batch=50, repeat=1)

                valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                      valid_config.step_size, shuffle=False, intra_shuffle=False)
                model.eval()
                valid_loss, loss_names = model.valid_model("ELBO_VALID", valid_feed, use_profile=config.use_profile)
                model.train()

                done_epoch = epoch + 1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if config.op == "sgd" and done_epoch > config.lr_hold:
                    model.learning_rate_decay()

                if valid_loss[0] < best_dev_loss:
                    if valid_loss[0] <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss[0]

                    # still save the best train model
                    if FLAGS.save_model:
                        print("Save model!!")
                        torch.save(model.state_dict(), dm_checkpoint_path %(epoch))
                    best_dev_loss = valid_loss[0]
                    best_dev_losses = valid_loss

                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
            print("Best validation loss %f" % best_dev_loss)
            model.print_loss("ELBO_BEST", loss_names,
                                     best_dev_losses, "")
            print("Done training")
        else:
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            model.eval()
            dest_f = open(os.path.join(log_dir, "test.txt"), "wb")
            model.test_model( valid_feed, repeat=10, dest=dest_f, use_profile=config.use_profile)

            model.train()
            dest_f.close()

if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()