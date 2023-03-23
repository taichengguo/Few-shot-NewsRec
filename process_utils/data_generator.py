import numpy as np
import re
import random
from collections import Counter
from itertools import *
from tqdm import tqdm
import pickle

def load_obj(name, save_root):
    with open(save_root + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class input_data(object):
    def __init__(self, args, args_mind=None):
        self.args = args
        self.args_mind = args_mind
        a_p_list_train = [[] for k in range(self.args.A_n)]
        relation_f = ["u_n_list_train.txt"]
        # store academic relational data
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(self.args.data_path + f_name, "r")
            for line in neigh_f:
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                if neigh_list_id[0] == '':
                    a_p_list_train[node_id] = []
                    continue

                for j in range(len(neigh_list_id)):
                    a_p_list_train[node_id].append('p' + str(neigh_list_id[j]))
            neigh_f.close()
        self.a_p_list_train = a_p_list_train

        p_title_embed_multi = []
        if self.args.db == 'mind':
            p_title_embed_multi = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)
            p_t_e_f = open(self.args.data_path + "n_title_embed_train.txt", "r")
            for line in islice(p_t_e_f, 1, None):
                values = line.split()
                index = int(values[0])
                embeds = np.asarray(values[1:], dtype='int32')
                p_title_embed_multi[index] = embeds
            p_t_e_f.close()

        elif self.args.db == 'adressa':
            if self.args.few_shot_method == 2:
                # translated
                if self.args.range == 'Model/data':
                    plm_encodes_path, plm_encodes_file = self.args_mind.data_path, 'n_origin_title_switched_0_embed_train.txt'
                else:
                    plm_encodes_path, plm_encodes_file = self.args_mind.data_path, 'n_origin_title_switched_0_{}_embed_train.txt'.format(self.args.target_domain_sim)

                if self.args.target_domain_sim == 0:
                    for data_path, adressa_token_path in [
                        (self.args.data_path, "n_title_embed_train.txt"),
                        (self.args.data_path, 'n_title_translated_embed_train.txt')
                    ]:
                        tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)

                        p_t_e_f = open(data_path + adressa_token_path, "r")
                        for line in islice(p_t_e_f, 1, None):
                            values = line.split()
                            index = int(values[0])
                            embeds = np.asarray(values[1:], dtype='int32')
                            tem[index] = embeds
                        p_t_e_f.close()
                        p_title_embed_multi.append(tem)
                else:
                    top2_file = 'n_origin_title_switched_1_{}_embed_train.txt'.format(self.args.target_domain_sim)
                    top3_file = 'n_origin_title_switched_2_{}_embed_train.txt'.format(self.args.target_domain_sim)
                    top4_file = 'n_origin_title_switched_3_{}_embed_train.txt'.format(self.args.target_domain_sim)
                    top5_file = 'n_origin_title_switched_4_{}_embed_train.txt'.format(self.args.target_domain_sim)
                    if self.args.topn == 2:
                        for data_path, adressa_token_path in [
                            (self.args.data_path, "n_title_embed_train.txt"),
                            (self.args.data_path, 'n_title_translated_embed_train.txt'),
                            (plm_encodes_path, plm_encodes_file),
                            (plm_encodes_path, top2_file)]:
                            tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)

                            p_t_e_f = open(data_path + adressa_token_path, "r")
                            for line in islice(p_t_e_f, 1, None):
                                values = line.split()
                                index = int(values[0])
                                embeds = np.asarray(values[1:], dtype='int32')
                                tem[index] = embeds
                            p_t_e_f.close()
                            p_title_embed_multi.append(tem)

                    elif self.args.topn == 3:
                        for data_path, adressa_token_path in [
                            (self.args.data_path, "n_title_embed_train.txt"),
                            (self.args.data_path, 'n_title_translated_embed_train.txt'),
                            (plm_encodes_path, plm_encodes_file),
                            (plm_encodes_path, top2_file),
                            (plm_encodes_path, top3_file)]:
                            tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)

                            p_t_e_f = open(data_path + adressa_token_path, "r")
                            for line in islice(p_t_e_f, 1, None):
                                values = line.split()
                                index = int(values[0])
                                embeds = np.asarray(values[1:], dtype='int32')
                                tem[index] = embeds
                            p_t_e_f.close()
                            p_title_embed_multi.append(tem)

                    elif self.args.topn == 4:
                        for data_path, adressa_token_path in [
                            (self.args.data_path, "n_title_embed_train.txt"),
                            (self.args.data_path, 'n_title_translated_embed_train.txt'),
                            (plm_encodes_path, plm_encodes_file),
                            (plm_encodes_path, top2_file),
                            (plm_encodes_path, top3_file),
                            (plm_encodes_path, top4_file)]:
                            tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)

                            p_t_e_f = open(data_path + adressa_token_path, "r")
                            for line in islice(p_t_e_f, 1, None):
                                values = line.split()
                                index = int(values[0])
                                embeds = np.asarray(values[1:], dtype='int32')
                                tem[index] = embeds
                            p_t_e_f.close()
                            p_title_embed_multi.append(tem)

                    elif self.args.topn == 5:
                        for data_path, adressa_token_path in [
                            (self.args.data_path, "n_title_embed_train.txt"),
                            (self.args.data_path, 'n_title_translated_embed_train.txt'),
                            (plm_encodes_path, plm_encodes_file),
                            (plm_encodes_path, top2_file),
                            (plm_encodes_path, top3_file),
                            (plm_encodes_path, top4_file),
                            (plm_encodes_path, top5_file)]:
                            tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)

                            p_t_e_f = open(data_path + adressa_token_path, "r")
                            for line in islice(p_t_e_f, 1, None):
                                values = line.split()
                                index = int(values[0])
                                embeds = np.asarray(values[1:], dtype='int32')
                                tem[index] = embeds
                            p_t_e_f.close()
                            p_title_embed_multi.append(tem)
                    else:
                        for data_path, adressa_token_path in [
                            (self.args.data_path, "n_title_embed_train.txt"),
                            (self.args.data_path, 'n_title_translated_embed_train.txt'),
                            (plm_encodes_path, plm_encodes_file)]:
                            tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)

                            p_t_e_f = open(data_path + adressa_token_path, "r")
                            for line in islice(p_t_e_f, 1, None):
                                values = line.split()
                                index = int(values[0])
                                embeds = np.asarray(values[1:], dtype='int32')
                                tem[index] = embeds
                            p_t_e_f.close()
                            p_title_embed_multi.append(tem)

                # load kg news
                # ad_mind_dict = load_obj("kg_enhanced", self.args_mind.data_path)
                # ad_mind_dict[-1] = [np.zeros((self.args.max_title_token), dtype=np.int)]
                # p_title_embed_multi.append(ad_mind_dict)
            else:
                for adressa_token_path in ["n_title_embed_train.txt"]:
                    tem = np.zeros((self.args.P_n + 1, self.args.max_title_token), dtype=np.int)
                    p_t_e_f = open(self.args.data_path + adressa_token_path, "r")
                    for line in islice(p_t_e_f, 1, None):
                        values = line.split()
                        index = int(values[0])
                        embeds = np.asarray(values[1:], dtype='int32')
                        tem[index] = embeds
                    p_t_e_f.close()
                    p_title_embed_multi.append(tem)

        self.p_body_embed = None

        self.p_abstract_embed = None
        self.p_title_embed = None
        self.p_title_embedmulti = p_title_embed_multi

        self.p_v_for_embed = None
        self.p_category_embed = None

        # store top neighbor set (based on frequency) from random walk sequence
        a_neigh_list_train_top = [[] for k in range(self.args.A_n)]
        for i in range(self.args.A_n):
            neigh_size = 50
            a_neigh_list_train_top[i] = [-1] * (neigh_size - len(self.a_p_list_train[i])) + \
                [int(p[1:]) for p in self.a_p_list_train[i]][-neigh_size:]

        self.a_neigh_list_train = a_neigh_list_train_top
