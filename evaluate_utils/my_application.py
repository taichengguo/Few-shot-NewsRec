#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 3:12 下午
# @Author  : taicheng.guo
# @Email:  : 2997347185@qq.com
# @File    : my_application.py


import re
from itertools import *
import torch
from args import read_args
torch.set_num_threads(2)
from evaluate_utils.evaluate import *
import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import wandb

"""
a_p_recommendation

train - run - metrics
"""

class recommend_evaluator(object):

    def __init__(self, args, iter):
        super(recommend_evaluator, self).__init__()
        self.args = args
        self.gpu = args.cuda
        self.iter = iter
        self.embed_d = args.embed_d

    def a_p_recommendation(self, a_embed, p_embed, a_embed2=None, model=None, best_valid_auc=None):
        if self.args.db == 'mind':
            if best_valid_auc is None:
                auc = self._recomend_by_filetag(a_embed, p_embed, 'valid', a_embed2, model)
            else:
                auc = self._recomend_by_filetag(a_embed, p_embed, 'valid', a_embed2, model)
                if auc >= best_valid_auc:
                    test_auc = self._recomend_by_filetag(a_embed, p_embed, 'test', a_embed2, model)
        else:
            auc = self._recomend_by_filetag(a_embed, p_embed, 'test', a_embed2, model)
        return auc

    def _recomend_by_filetag(self, a_embed, p_embed, tag, a_embed2=None, model=None):
        tc = pd.read_csv(self.args.data_path + "{}_samples.txt".format(tag))
        predictions = []
        import ast
        from tqdm import tqdm

        for idx, row in tqdm(tc.iterrows()):
            uid = row[0]
            # record limited uids
            impre_id = row[1]
            candidates = ast.literal_eval(row[2])
            candidates_sim = [np.dot(a_embed[uid], p_embed[candidate]) for candidate in candidates]
            candidates_rank = [sorted(candidates_sim)[::-1].index(x)+1 for x in candidates_sim]
            predictions.append((impre_id, candidates_rank))

        auc = self._eval_auc(predictions, tag)

        return auc

    def _eval_auc(self, predictions, tag, prefix=''):

        pred_file_name = "./tt" + self.args.few_shot + "-" + str(self.args.few_shot_method) + \
                         "-" + str(self.args.news_cls_iter) + "-" + str(self.args.target_domain_sim) + "-" + \
                         str(self.args.loss_weight) + "-" + str(self.args.random_seed) + "-" + \
                         str(self.args.loss_weight_align) + "-" + str(tag) + \
                         str(self.args.range).replace("/", "-") + ".txt"

        pred_f = open(pred_file_name, 'w')
        for idx, atuple in enumerate(predictions):
            if idx == len(predictions) - 1:
                pred_f.write(str(atuple[0]) + " " + str(atuple[1]).replace(" ", ""))
            else:
                pred_f.write(str(atuple[0]) + " " + str(atuple[1]).replace(" ", "") + "\n")
        pred_f.close()

        true_file = open(self.args.data_path + "{}_labels.txt".format(tag), 'r')
        sub_file = open(pred_file_name, 'r')

        auc, mrr, ndcg, ndcg10, f1 = scoring(true_file, sub_file, db=self.args.db, limit_impressions=None, tag=tag)
        if self.args.db == 'adressa':
            print("{} AUC:{:.4f}\tMRR:{:.4f}\tnDCG@5:{:.4f}\tnDCG@10:{:.4f}".format(tag, auc, mrr, ndcg, ndcg10))
            wandb.log({"adressa_{}_auc".format(tag): auc})
        else:
            print("{} - {} AUC:{:.4f}\tMRR:{:.4f}\tnDCG@5:{:.4f}\tnDCG@10:{:.4f}".format(prefix, tag, auc, mrr, ndcg,
                                                                                         ndcg10))
            wandb.log({"mind_{}-{}_auc".format(prefix, tag): auc})
            wandb.log({"mind_{}-{}_mrr".format(prefix, tag): mrr})
            wandb.log({"mind_{}-{}_ndcg@5".format(prefix, tag): ndcg})
            wandb.log({"mind_{}-{}_ndcg@10".format(prefix, tag): ndcg10})
        return auc

    def fit(self, train_samples):

        X = train_samples[:, 0:2]
        y = train_samples[:, 2]
        clf = LogisticRegressionCV(random_state=42, cv=5)
        clf.fit(X, y)
        print(clf.coef_)
        return clf

    def predict(self, predictions_scores):
        # out: candidate_rank
        predictions = []

        for idx, atuple in enumerate(predictions_scores):
            sim_normal = [float(i)/sum(atuple[1]) for i in atuple[1]]
            sim2_normal = [float(i)/sum(atuple[2]) for i in atuple[2]]
            predict_samples = list(zip(sim_normal, sim2_normal))
            predict_samples = np.array(predict_samples)

            probas = list(self.clf.predict_proba(predict_samples)[:, 1])
            candidates_rank = [sorted(probas)[::-1].index(x)+1 for x in probas]
            predictions.append((atuple[0], candidates_rank))

        return predictions

    def construct_train_samples(self, predictions_scores):

        train_samples = []

        true_file = self.args.data_path + "{}_labels.txt".format('valid')
        label_df = pd.read_csv(true_file, header=None, sep=' ')
        label_df.columns = ['impre_id', 'label']
        label_df['label'] = label_df['label'].apply(lambda x: ast.literal_eval(x))

        for idx, atuple in enumerate(predictions_scores):
            sim_normal = [float(i)/sum(atuple[1]) for i in atuple[1]]
            sim2_normal = [float(i)/sum(atuple[2]) for i in atuple[2]]
            labels = label_df.iloc[idx]['label']
            tem = list(zip(sim_normal, sim2_normal, labels))
            train_samples = train_samples + tem

        train_samples = np.array(train_samples)
        return train_samples

