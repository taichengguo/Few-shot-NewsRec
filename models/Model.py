#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 11:20 下午
# @Author  : taicheng.guo
# @Email:  : 2997347185@qq.com
# @File    : HetGNN.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from args import read_args
import numpy as np
import math
args = read_args()
from bpemb import BPEmb
from layers.attention import AdditiveAttention
import random
from tqdm import tqdm
import copy
import pandas as pd
import pdb

class RecModel(nn.Module):
    def __init__(self, args, feature_list, a_neigh_list_train, u_n_train_dict=None, lang='en'):
        super(RecModel, self).__init__()
        embed_d = 300
        self.args = args
        self.P_n = args.P_n
        self.A_n = args.A_n
        self.V_n = args.V_n
        self.entity_n = args.entity_n

        self.feature_list = feature_list

        self.a_train_id_list = list(range(args.A_n))
        self.p_train_id_list = list(range(args.P_n))

        self.a_neigh_list_train = a_neigh_list_train

        # #### Adressa数据集infer阶段，增加train的news构图 ####
        # # update self.a_neight_list_train_for_test
        # if u_n_train_dict is not None:
        #     self.a_neight_list_train_for_test = copy.deepcopy(a_neigh_list_train)
        #     for uid, news_list in u_n_train_dict.items():
        #         self.a_neight_list_train_for_test[uid] += news_list
        #         self.a_neight_list_train_for_test[uid] = self.a_neight_list_train_for_test[uid][-50:]

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.embed_d = embed_d

        # using bpe token
        self.lang = lang

        bpeEmb_multi = BPEmb(lang="multi", dim=300, vs=320000)
        self.word_emb_layer = nn.Embedding.from_pretrained(tensor(bpeEmb_multi.vectors), freeze=False, padding_idx=0)
        self.title_multi_self_attention = nn.MultiheadAttention(embed_d, num_heads=20)
        self.title_multi_attention = AdditiveAttention(300, 200)

        self.user_attention = AdditiveAttention(300, 200)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def update_token_cs(self, p_title_new_token):
        """
        Returns:
        """
        # print("upadte")
        self.feature_list[5] = p_title_new_token
        # print(type(self.feature_list[5]))

    def change_config(self, args, feature_list, a_neigh_list_train, u_n_train_dict=None):
        self.args = args
        self.P_n = args.P_n
        self.A_n = args.A_n
        self.V_n = args.V_n
        self.feature_list = feature_list

        self.a_train_id_list = list(range(args.A_n))
        self.p_train_id_list = list(range(args.P_n))
        self.a_neigh_list_train = a_neigh_list_train

        #### Adressa数据集infer阶段，增加train的news构图 ####
        # update self.a_neight_list_train_for_test
        if u_n_train_dict is not None:
            self.a_neight_list_train_for_test = copy.deepcopy(a_neigh_list_train)
            for uid, news_list in u_n_train_dict.items():
                self.a_neight_list_train_for_test[uid] += news_list
                self.a_neight_list_train_for_test[uid] = self.a_neight_list_train_for_test[uid][-50:]

    def generate_mask(self, a_p_batch, padding_tag=-1):
        # 0 -> True; 1 -> False
        mask_list = []
        for useq in a_p_batch:
            if sum(useq) == 0:
                tem = [False] * len(useq)
            else:
                tem = [True if i == padding_tag else False for i in useq]
            mask_list.append(tem)
        mask_tensor = torch.from_numpy(np.array(mask_list))
        if args.cuda:
            mask_tensor = mask_tensor.cuda()
        return mask_tensor


    def a_content_agg(self, id_batch, mode='train', cross_a_neigh_list_train=None, cross_model=None):
        id_batch = np.reshape(id_batch, (1, -1))
        batch_s = id_batch.shape[1]

        # a neighbors p => all p title self attention
        # user encoder: a最近20/50个news的
        a_p_batch = [self.a_neigh_list_train[a] for a in id_batch[0]]  # [B*1*T]
        mask_tensor = self.generate_mask(a_p_batch)     # B*seq
        p_num = len(a_p_batch[0])
        a_p_batch = np.reshape(a_p_batch, (1, -1))

        a_p_embedding = self.p_content_agg(a_p_batch)
        a_p_embedding = a_p_embedding.view(batch_s, p_num, self.embed_d)   # [B * seq * embed]

        weight_user_batch, _ = self.user_attention(a_p_embedding, mask_tensor)
        return weight_user_batch

    def news_token_random_generator(self, id_batch):
        """
        Mix up
        Returns:
        """
        p_title_embed_multi = self.feature_list[5]
        tokens = []
        for id in id_batch:
            tem = random.sample(p_title_embed_multi, 1)
            tem_token = tem[id]
            tokens.append(tem_token)
        tokens = torch.from_numpy(np.array(tokens)).cuda()
        return tokens

    def p_content_agg(self, id_batch, mode='train'):
        id_batch = np.reshape(id_batch, (1, -1))
        batch_s = id_batch.shape[1]
        """
        word - embedding - self attention - attention
        id_batc 
        """
        # title encoder multi embedding
        if self.args.db == 'mind':
            print(1)
            p_t_multi_token = self.feature_list[5][id_batch]
        else:
            p_t_multi_token = self.news_token_random_generator(id_batch)

        p_t_multi_batch = self.drop(self.word_emb_layer(p_t_multi_token))  # [B * token * emb]
        mask_tensor_multi = self.generate_mask(p_t_multi_token.tolist(), padding_tag=0)

        p_t_multi_batch = p_t_multi_batch.permute(1, 0, 2)
        p_t_multi_batch, _ = self.title_multi_self_attention(p_t_multi_batch, p_t_multi_batch,
                                                            p_t_multi_batch,
                                                            key_padding_mask=mask_tensor_multi)  # [token * B * emb]
        p_t_multi_batch = self.drop(p_t_multi_batch.permute(1, 0, 2))
        weight_title_batch, _ = self.title_multi_attention(p_t_multi_batch, mask_tensor_multi)  # [B * emb]
        return weight_title_batch

    def node_het_agg(self, id_batch, node_type, mode='train'):  # heterogeneous neighbor aggregation
        # attention module
        # id_batch = np.reshape(id_batch, (1, -1))
        if node_type == 1:
            c_agg_batch = self.a_content_agg(id_batch, mode=mode)
        elif node_type == 2:
            c_agg_batch = self.p_content_agg(id_batch)

        weight_agg_batch = c_agg_batch
        # do normalization
        # TODO
        # weight_agg_batch = F.normalize(weight_agg_batch, p=2, dim=1)

        return weight_agg_batch

    def het_agg(self, triple_index, c_id_batch, neg_id_batch, model=None):
        embed_d = self.embed_d
        # batch processing
        mapper = [(1, 2)]

        if triple_index < 1:
            c_agg = self.node_het_agg(c_id_batch, mapper[triple_index][0], model)
            n_agg = self.node_het_agg(neg_id_batch, mapper[triple_index][1], model)

        if triple_index == 16:  # save learned node embedding and do test: predict
            a_embed = np.around(np.random.normal(0, 0.01, [self.args.A_n, self.args.embed_d]), 4)
            p_embed = np.around(np.random.normal(0, 0.01, [self.args.P_n, self.args.embed_d]), 4)

            with torch.no_grad():
                save_batch_s = self.args.mini_batch_s
                for i in range(2):
                    if i == 0:
                        batch_number = int(len(self.a_train_id_list) / save_batch_s)
                    elif i == 1:
                        batch_number = int(len(self.p_train_id_list) / save_batch_s)

                    for j in tqdm(range(batch_number)):
                        if i == 0:
                            id_batch = self.a_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                            if self.args.db == 'adressa':
                                out_temp = self.node_het_agg(id_batch, 1, mode='test')
                            else:
                                out_temp = self.node_het_agg(id_batch, 1, model)
                        elif i == 1:
                            id_batch = self.p_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                            out_temp = self.node_het_agg(id_batch, 2)

                        out_temp = out_temp.data.cpu().numpy()

                        for k in range(len(id_batch)):
                            index = id_batch[k]
                            if i == 0:
                                a_embed[index] = out_temp[k]
                            elif i == 1:
                                p_embed[index] = out_temp[k]

                    if i == 0:
                        id_batch = self.a_train_id_list[batch_number * save_batch_s:]
                        if len(id_batch) == 0:
                            continue
                        else:
                            if self.args.db == 'adressa':
                                out_temp = self.node_het_agg(id_batch, 1, mode='test')
                            else:
                                out_temp = self.node_het_agg(id_batch, 1, model)
                    elif i == 1:
                        id_batch = self.p_train_id_list[batch_number * save_batch_s:]
                        if len(id_batch) == 0:
                            continue
                        else:
                            out_temp = self.node_het_agg(id_batch, 2)

                    out_temp = out_temp.data.cpu().numpy()

                    for k in range(len(id_batch)):
                        index = id_batch[k]
                        if i == 0:
                            a_embed[index] = out_temp[k]
                        elif i == 1:
                            p_embed[index] = out_temp[k]
                return a_embed, p_embed

        return c_agg, n_agg

    def aggregate_all(self, triple_list_batch, triple_index, model=None):
        c_id_batch = [x[0] for x in triple_list_batch]
        tem = [x[1] for x in triple_list_batch]
        neg_id_batch = []

        for x in tem:
            neg_id_batch += x

        c_agg, pos_agg = self.het_agg(triple_index, c_id_batch, neg_id_batch, model)
        return c_agg, pos_agg

    def forward(self, triple_list_batch, triple_index, mode='single'):
        c_out, p_out = self.aggregate_all(triple_list_batch, triple_index, mode)
        return c_out, p_out