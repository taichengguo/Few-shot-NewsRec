# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from process_utils import data_generator, tools
from models import HetGNN
from torch.utils.data import DataLoader, RandomSampler
import random
torch.set_num_threads(2)
from evaluate_utils.my_application import *
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from process_utils.Iterator import MINDIterator
from config import hparams
import pickle
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from process_utils.MLMIter import AlignDatset
wandb.init(project="NewsRec", name="newsrec3")

class model_class(object):
    def __init__(self, args, args_mind):
        super(model_class, self).__init__()
        self.args = args
        self.gpu = args.cuda

        input_data = data_generator.input_data(self.args, args_mind)
        self.input_data = input_data
        if args.db == 'mind':
            self.train_behaviors_file = self.args.data_path + "behaviors.tsv"
        else:
            self.train_behaviors_file = self.args.data_path + "behaviors.tsv"
        self.save_root = self.args.data_path

        # CTR prediction
        self.train_iterator = MINDIterator(
            batch_size = self.args.mini_batch_s,
            npratio=self.args.npratio,
            col_spliter="\t",
        )

        if args.db == 'mind':
            feature_list = [input_data.p_title_embed, input_data.p_abstract_embed,
                            input_data.p_v_for_embed, input_data.p_category_embed,
                            input_data.p_body_embed, input_data.p_title_embedmulti]
        else:
            feature_list = [input_data.p_title_embed, input_data.p_abstract_embed,
                            input_data.p_v_for_embed, input_data.p_category_embed,
                            np.zeros((1,1)), input_data.p_title_embedmulti]

        if self.args.db == 'mind':
            feature_list[5] = torch.from_numpy(np.array(feature_list[5]))
            if self.gpu:
                feature_list[5] = feature_list[5].cuda()

        self.p_title_embedmulti = feature_list[5]

        if self.args.db == 'adressa':
            # get train behaviors + get train clicks + dict
            df = pd.read_csv(self.train_behaviors_file, sep='\t', header=None)
            df.columns = ['impre_id', 'user', 'time', 'his', 'impr']
            uid2union_id = self.load_obj("uid2union_id")
            news2union_id = self.load_obj("news2union_id")
            u_n_train = {}
            for idx, row in tqdm(df.iterrows()):
                user = uid2union_id[row['user']]
                impr = [i.split("-")[0] for i in row['impr'].split(" ") if i.split("-")[1] == '1']
                impr = [news2union_id[i] for i in impr]
                # append
                u_n_train[user] = u_n_train.get(user, []) + impr
            self.model = HetGNN.RecModel(args, feature_list,
                                         input_data.a_neigh_list_train,
                                         u_n_train)
        else:
            self.model = HetGNN.RecModel(args, feature_list,
                                         input_data.a_neigh_list_train)

        if self.gpu:
            self.model.cuda()

    # self.model.init_weights()

    def save_obj(self, obj, name):
        with open(self.save_root + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open(self.save_root + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def convert_data_for_hetGNN(self, batch_data_input):
        """
		Data to HetGNN
		Args:
			batch_data_input:

		Returns:
		"""
        ctr_samples = []

        candidate_raw = batch_data_input['candidate_raw']
        user_raw = batch_data_input['user_raw']

        uid2union_id = self.load_obj("uid2union_id")
        news2union_id = self.load_obj("news2union_id")
        news2union_id[-1] = -1  # padding

        for idx, uid in enumerate(user_raw):
            raw_news = candidate_raw[idx]
            ctr_samples.append((uid2union_id[uid], [news2union_id[i] for i in raw_news]))

        return ctr_samples

    def save_train_samples(self, ctr_samples):
        """
        Save train samples for train evaluation
        Args:
            ctr_samples:
        Returns: None
        """
        all_df = []
        labels = []
        impr = 0
        for sample in ctr_samples:
            uid = sample[0]
            news = [i for i in sample[1] if i != -1]
            all_df.append((uid, impr, news))
            labels.append((impr, [1] + [0] * (len(news)-1)))
            impr += 1
        pd.DataFrame(all_df, columns=['user_id', 'impression_id', 'impressions_candidate'])\
            .to_csv(self.args.data_path + "train_samples.txt", index=False)
        labels_df = pd.DataFrame(labels, columns=['impression_id', 'impressions_label'])
        label_file = open(self.args.data_path + "train_labels.txt", 'w')
        k = 0
        for idx, row in labels_df.iterrows():
            if k == len(labels_df) - 1:
                label_file.write(str(row['impression_id']) + " " + str(row['impressions_label']).replace(" ", ""))
            else:
                label_file.write(
                    str(row['impression_id']) + " " + str(row['impressions_label']).replace(" ", "") + "\n")
            k += 1
        label_file.close()

    def eval_rec(self):
        """
        eval rec task
        Returns:
        """
        pass

    def load_feature_list(self, input_data):
        if self.args.db == 'mind':
            feature_list = [input_data.p_title_embed, input_data.p_abstract_embed,
                            input_data.p_v_for_embed, input_data.p_category_embed,
                            input_data.p_body_embed, input_data.p_title_embedmulti]
        else:
            feature_list = [input_data.p_title_embed, input_data.p_abstract_embed,
                            input_data.p_v_for_embed, input_data.p_category_embed,
                            np.zeros((1,1)), input_data.p_title_embedmulti]
        if self.args.db == 'mind':
            feature_list[5] = torch.from_numpy(np.array(feature_list[5]))
            if self.gpu:
                feature_list[5] = feature_list[5].cuda()
        return feature_list

    def change_config(self, args, args_mind=None):
        self.args = args
        input_data = data_generator.input_data(self.args, args_mind)
        self.input_data = input_data
        self.train_behaviors_file = self.args.data_path + "behaviors.tsv"
        self.save_root = self.args.data_path
        feature_list = self.load_feature_list(input_data)
        self.model.change_config(args, feature_list, input_data.a_neigh_list_train, None)
        self.p_title_embedmulti = feature_list[5]

        self.train_iterator = MINDIterator(
            batch_size = self.args.mini_batch_s,
            npratio=self.args.npratio,
            col_spliter="\t",
        )

    def get_bilingual_dict(self):

        no2en = pd.read_csv("../../muse/no-en.txt", sep='\t', header=None)
        no2endict = {}
        for idx, row in no2en.iterrows():
            no2endict[row[0]] = no2endict.get(row[0], []) + [row[1]]
        return no2endict

    def cross(self, x, disable=False):
        if not disable and (self.token_rate >= random.random()):
            if x in self.no2endict:
                return self.no2endict[x][random.randint(0, len(self.no2endict[x]) - 1)]
            else:
                return x
        else:
            return x

    def cross_str(self, x, disable=False):
        raw = x.lower().split(" ")
        out = ""
        for xx in raw:
            out += self.cross(xx, disable)
            out += " "
        return out

    def re_encode_text(self, df):
        from bpemb import BPEmb
        import numpy as np
        bpemb_no = BPEmb(lang="multi", dim=300, vs=320000)
        max_title_token = 30
        p_title_embed = np.zeros((len(df) + 1, max_title_token), dtype=np.int)

        def bpe_encode_text(x, bpemb_en, max_title_size):
            if isinstance(x, str):
                tokens = bpemb_en.encode_ids(x)
            else:
                tokens = []
            return tokens[:max_title_size] + [0] * (max_title_size - len(tokens))

        for idx, row in df.iterrows():
            title = row['title']
            encode = bpe_encode_text(title, bpemb_no, max_title_token)
            embeds = np.asarray(encode, dtype='int32')
            p_title_embed[row['nid']] = embeds

        return p_title_embed


    def add_code_switch(self, sen_rate, token_rate):
        """
        all news titles -> replace -> encode -> update
        Args:
            sen_rate:
            token_rate:

        Returns:
        """
        self.sen_rate = sen_rate
        self.token_rate = token_rate
        self.no2endict = self.get_bilingual_dict()

        ad_news2id = self.load_obj('news2union_id')
        ad_newsid2title = pd.read_csv(self.args.data_path + "newsid2title.csv")
        docid2title = {}
        for idx, row in ad_newsid2title.iterrows():
            docid2title[row['doc_id']] = row['title']
        nid2title = {}
        for docid, nid in ad_news2id.items():
            nid2title[nid] = docid2title[docid]
        nid2titledf = pd.DataFrame(nid2title.items())
        nid2titledf.columns = ['nid', 'title']

        nid2titledf['title_cross'] = nid2titledf['title'].apply(
            lambda x: self.cross_str(x, not (sen_rate >= random.random())))

        p_title_embed = self.re_encode_text(nid2titledf)
        p_title_embed = torch.from_numpy(np.array(p_title_embed))
        if self.gpu:
            p_title_embed = p_title_embed.cuda()
        self.model.update_token_cs(p_title_embed)

    def add_code_switchbynews(self, news_rate, news2token):
        """
        replace news with new token
        Args:
            sen_rate:
            token_rate:

        Returns:
        """
        # read adressa news id -> same subcategory english news id
        adid2mind = self.load_obj('adid2mind')
        for i in range(self.args.P_n):
            if i in adid2mind:
                # judge
                if news_rate > random.random():
                    # search and replace
                    mind_news_id = random.choice(adid2mind[i])
                    token = news2token[mind_news_id]
                    # print(type(news2token))
                    # print(type(token))
                    self.p_title_embedmulti[i] = token
        self.model.update_token_cs(self.p_title_embedmulti)


if __name__ == '__main__':

    best_valid_auc = 0
    # lr depends on transfer setting
    args_mind = read_args(db='mind', lr=3e-4)
    args_ad = read_args(db='adressa', lr=3e-4)

    if args_mind.range != 'Model/engTonor':
        print("set lr")
        args_mind.lr = 1e-4
        args_ad.lr = 1e-4

    print("few shot method is {}".format(args_mind.few_shot_method))

    print("------arguments-------")
    for k, v in vars(args_mind).items():
        print(k + ': ' + str(v))
    for k, v in vars(args_ad).items():
        print(k + ': ' + str(v))

    # fix random seed
    random.seed(args_mind.random_seed)
    np.random.seed(args_mind.random_seed)
    torch.manual_seed(args_mind.random_seed)
    torch.cuda.manual_seed_all(args_mind.random_seed)

    # model + different lr
    model_mind = model_class(args_ad, args_mind)
    embed_d = args_ad.embed_d

    parameters = list(model_mind.model.parameters())
    optimizer_mind = optim.Adam(parameters, lr=args_mind.lr, weight_decay=1e-8)

    for iter_i in range(args_ad.train_iter_n):
        model_mind.change_config(args_ad, args_mind)
        model_mind.model.train()
        print('epoch ' + str(iter_i) + ' ...' + "lr is {}".format(optimizer_mind.param_groups[0]['lr']))

        if args_mind.few_shot_method in [2]:
            # news classification
            model_mind.change_config(args_ad, args_mind)
            model_mind.model.train()
            loss_record = 0
            newsids = list(range(args_ad.P_n))
            bz = 64
            for i in range(args_mind.news_cls_iter):
                for i in range(0, len(newsids), bz):
                    tem_ids = newsids[i: i+bz]
                    # TODO
                    loss = args_mind.loss_weight_align * model_mind.model.news_align_loss(tem_ids)
                    optimizer_mind.zero_grad()
                    loss.backward()
                    optimizer_mind.step()
                    loss_record += 1
                    if loss_record % 30 == 0:
                        print("news cls loss: {}".format(loss))

        if args_mind.few_shot_method in [1,2]:
            # adressa training samples
            model_mind.change_config(args_ad, args_mind)
            model_mind.model.train()
            loss_record = 0
            for mind_data_input in model_mind.train_iterator.load_data_from_file(model_mind.train_behaviors_file):
                ctr_samples = model_mind.convert_data_for_hetGNN(mind_data_input)
                c_out, p_out = model_mind.model(ctr_samples, 0)
                # TODO
                loss_mind = args_mind.loss_weight * tools.cross_entropy_loss(c_out, p_out, embed_d)
                optimizer_mind.zero_grad()
                loss_mind.backward()
                optimizer_mind.step()
                loss_record += 1
                if loss_record % 60 == 0:
                    print("ad loss: {}".format(loss_mind))
                    wandb.log({"ad_loss": loss_mind})

        if args_mind.few_shot.split("_")[-1] != '0shot':
            loss_record = 0
            # Few-shot setting: mind training samples
            model_mind.change_config(args_mind, args_mind)
            model_mind.model.train()
            for mind_data_input in model_mind.train_iterator.load_data_from_file(model_mind.train_behaviors_file):
                ctr_samples = model_mind.convert_data_for_hetGNN(mind_data_input)
                c_out, p_out = model_mind.model(ctr_samples, 0)
                loss_mind = tools.cross_entropy_loss(c_out, p_out, embed_d)
                optimizer_mind.zero_grad()
                loss_mind.backward()
                optimizer_mind.step()
                loss_record += 1
                if loss_record % 1 == 0:
                    print("mind loss: {}".format(loss_mind))
                    wandb.log({"mind_loss": loss_mind})

        # evaluation adressa
        # model_mind.model.eval()
        # a_embed, p_embed = model_mind.model([], 16)
        # r = recommend_evaluator(model_mind.args, iter=iter_i)
        # mind_auc = r.a_p_recommendation(a_embed, p_embed, None, model_mind)

        ############################   ############################
        # evaluation
        ############################   ############################
        # if iter_i % 3 == 0:
        #     print("Start evaluating...")
        model_mind.change_config(args_mind, args_mind)
        model_mind.model.eval()
        a_embed, p_embed = model_mind.model([], 16)
        r = recommend_evaluator(model_mind.args, iter=iter_i)
        val_auc = r.a_p_recommendation(a_embed, p_embed, None, model_mind, best_valid_auc)
        if val_auc >= best_valid_auc:
            best_valid_auc = val_auc

            if model_mind.args.save_emb:
                # save mind embedding
                model_mind.save_obj(a_embed, "vis_mind_user_embed2")
                model_mind.save_obj(p_embed, 'vis_mind_news_embed2')

                # save adressa embedding
                model_mind.change_config(args_ad, args_mind)
                model_mind.model.eval()
                a_embed, p_embed = model_mind.model([], 16, mode='test')
                model_mind.save_obj(a_embed, 'vis_ad_user_embed2')
                model_mind.save_obj(p_embed, 'vis_ad_news_embed2')

    ############################   ############################
    # evaluation
    ############################   ############################

    # print("Start evaluating...")
    # model_mind.model.eval()
    # a_embed, p_embed = model_mind.model([], 16, mode='test')
    # r = recommend_evaluator(model_mind.args, iter=args_ad.train_iter_n - 1)
    # mind_auc = r.a_p_recommendation(a_embed, p_embed, None, model_mind)

    # save adressa embedding
    # if model_mind.args.save_emb:
    #     model_mind.save_obj(a_embed, 'vis_ad_user_embed')
    #     model_mind.save_obj(p_embed, 'vis_ad_news_embed')
    #
    # model_mind.change_config(args_mind, args_mind)
    # triple_index = 16
    # model_mind.model.eval()

    # a_embed, p_embed = model_mind.model([], triple_index)
    #
    # # save mind embedding
    # if model_mind.args.save_emb:
    #     model_mind.save_obj(a_embed, "vis_mind_user_embed")
    #     model_mind.save_obj(p_embed, 'vis_mind_news_embed')
    #
    # r = recommend_evaluator(model_mind.args, iter=args_ad.train_iter_n-1)
    # mind_auc = r.a_p_recommendation(a_embed, p_embed, None, model_mind)

























