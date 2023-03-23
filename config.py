#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/2 4:00 下午
# @Author  : taicheng.guo
# @Email:  : 2997347185@qq.com
# @File    : config.py

from args import read_args

args = read_args()

hparams = {

    'title_size': 30,
    'his_size': 50,

    'npratio': 4,
    'word_emb_dim': 300,

    # model
    'encoder_size': 300,

    'v_size': 200,
    'embed_size': 300,
    'nhead': 20,

    'batch_size': args.mini_batch_s,
    'epochs': 20,
    'learning_rate': 0.0005,

    'wordDict_file': './data/process_utils/word_dict.pkl',
    'userDict_file': './data/process_utils/uid2index.pkl',
    'wordEmb_file':  './data/process_utils/embedding.npy',

}