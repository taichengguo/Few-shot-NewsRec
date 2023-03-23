#!/usr/bin/env python
import numpy as np
import json
from sklearn.metrics import roc_auc_score, f1_score
# from args import read_args

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def parse_line(l):
    try:
        impid, ranks = l.strip('\n').split()
    except:
        print(l)
    ranks = json.loads(ranks)
    return impid, ranks

def scoring(truth_f, sub_f, db='mind', limit_impressions=None, tag=None):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    f1s = []
    impids = []
    
    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)
        impids.append(impid)

        # ignore masked impressions
        if labels == []:
            continue 
        
        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))

        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))        

        if limit_impressions is not None:
            if int(sub_impid) not in limit_impressions:
                continue

        lt_len = float(len(labels))

        # 更改为sigmoid
        y_true =  np.array(labels,dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1./rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)

        auc = roc_auc_score(y_true,y_score)
        mrr = mrr_score(y_true,y_score)
        ndcg5 = ndcg_score(y_true,y_score,5)
        ndcg10 = ndcg_score(y_true,y_score,10)
        
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)

        line_index += 1

    import pandas as pd
    # save each sample auc
    # if db == 'mind':
    #     test_sample_auc = pd.DataFrame({"impression_id": impids, "auc": aucs})
    #     test_sample_auc.to_csv("user_group/mind_each_aucs_{}_{}".format(np.mean(aucs), tag), index=False)

    # auc_all = roc_auc_score(y_label_all, y_score_all)
    import pandas as pd
    if db == 'adressa':
        # test_sample_auc = pd.DataFrame({"impid": impids, "auc": aucs})
        # test_sample_auc.to_csv("user_group/ad_each_aucs_{}_{}".format(np.mean(aucs), tag), index=False)
        return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s), None
    # if db == 'mind' and mode == 'test':
    #     file_appendix = str(round(float(np.mean(aucs)), 4))
        # pd.DataFrame(aucs).to_csv("aucs_{}.csv".format(file_appendix), index=False, header=None)
    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s), None


# if __name__ == '__main__':
#     args = read_args()
#
#     true_file = open(args.data_path + "test_labels.txt", 'r')
#     sub_file = open(args.data_path + "preds.txt", 'r')
#
#     auc, mrr, ndcg, ndcg10 = scoring(true_file, sub_file)
#
#     print("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))
