#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import ProfileBaselineConfig as Config
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader
from models.cvae import KgRnnCVAE

import torch

import glob

# constants
tf.app.flags.DEFINE_string("word2vec_path", None, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data/full_swda_clean_42da_sentiment_dialog_corpus.p", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "run1500783422", "the dir to load checkpoint for forward only")
FLAGS = tf.app.flags.FLAGS

def get_checkpoint_state(ckp_dir):
    files = os.path.join(ckp_dir, "*.pth")
    files = glob.glob(files)
    files.sort(key=os.path.getmtime)
    return len(files) > 0 and files[-1] or None

def main():
    # config for training
    config = Config()

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

    pp(config)

    # get data set