#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/2 3:49 下午
# @Author  : taicheng.guo
# @Email:  : 2997347185@qq.com
# @File    : Iterator.py

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pickle
import random
import re
random.seed(42)

def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """ Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.
    Args:
        news (list): input news list
        ratio (int): sample number
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [-1] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


__all__ = ["MINDIterator"]


class MINDIterator(object):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts.

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def __init__(
            self, batch_size, npratio=-1, col_spliter="\t", ID_spliter="%",
    ):
        """Initialize an iterator. Create necessary placeholders for the model.

        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = batch_size
        # self.title_size = hparams['title_size']
        # self.his_size = hparams['his_size']
        self.npratio = npratio

        self.word_dict = {}
        self.uid2index = {}

    def load_dict(self, file_path):
        """ load pickle file
        Args:
            file path (str): file path

        Returns:
            (obj): pickle load obj
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def init_news(self, news_file):
        """ init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """

        self.nid2index = {}
        news_title = [""]

        with open(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = word_tokenize(title)
                news_title.append(title)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        # for news_index in range(len(news_title)):
        #     title = news_title[news_index]
        #     for word_index in range(min(self.title_size, len(title))):
        #         if title[word_index] in self.word_dict:
        #             self.news_title_index[news_index, word_index] = self.word_dict[
        #                 title[word_index].lower()
        #             ]

    def init_behaviors(self, behaviors_file):
        """ init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []
        self.uids = []
        self.impre_real_news = []

        with open(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                impr_real_new = [i.split("-")[0] for i in impr.split()]

                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                self.uids.append(uid)
                self.impre_real_news.append(impr_real_new)
                impr_index += 1

    def parser_one_line(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negtive sampled result.

        Args:
            line (int): sample index.

        Returns:
            list: Parsed results including label, impression id , user id,
            candidate_title_index, clicked_title_index.
        """
        if self.npratio > 0:
            impr_label = self.labels[line]
            impre_real = self.impre_real_news[line]
            uid = self.uids[line]

            poss_raw = []
            negs_raw = []

            for click, raw_new in zip(impr_label, impre_real):
                if click == 1:
                    poss_raw.append(raw_new)
                else:
                    negs_raw.append(raw_new)

            for idx, p in enumerate(poss_raw):
                n_raw = newsample(negs_raw, self.npratio)
                candidate_raw_index = [poss_raw[idx]] + n_raw

                yield (
                    candidate_raw_index,
                    uid
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_title_index.append(self.news_title_index[news])
                click_title_index = self.news_title_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                )

    def load_data_from_file(self, behavior_file):
        """Read and parse data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed results, in the format of dict.
        """

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_title_indexes = []
        click_title_indexes = []
        cnt = 0

        candidate_raw_indexes = []
        user_raw_indexes = []

        indexes = np.arange(len(self.labels))

        # if self.npratio > 0:
        #     np.random.shuffle(indexes)

        for index in indexes:
            for (
                    candidate_raw_index,
                    uid

            ) in self.parser_one_line(index):

                candidate_raw_indexes.append(candidate_raw_index)
                user_raw_indexes.append(uid)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(

                        candidate_raw_indexes,
                        user_raw_indexes,
                    )
                    candidate_raw_indexes = []
                    user_raw_indexes = []

                    cnt = 0

        if cnt > 0:
            yield self._convert_data(
                candidate_raw_indexes,
                user_raw_indexes,
            )

    def _convert_data(
            self,
            candidate_raw_indexes,
            user_raw_indexes,

    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        return {
            "candidate_raw": candidate_raw_indexes,
            "user_raw": user_raw_indexes,
        }

    def load_user_from_file(self, news_file, behavior_file):
        """Read and parse user data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed user feature, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        user_indexes = []
        impr_indexes = []
        click_title_indexes = []
        cnt = 0

        for index in range(len(self.impr_indexes)):
            click_title_indexes.append(self.news_title_index[self.histories[index]])
            user_indexes.append(self.uindexes[index])
            impr_indexes.append(self.impr_indexes[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_user_data(
                    user_indexes, impr_indexes, click_title_indexes,
                )
                user_indexes = []
                impr_indexes = []
                click_title_indexes = []
                cnt = 0

        if cnt > 0:
            yield self._convert_user_data(
                user_indexes, impr_indexes, click_title_indexes,
            )

    def _convert_user_data(
            self, user_indexes, impr_indexes, click_title_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            user_indexes (list): a list of user indexes.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        impr_indexes = np.asarray(impr_indexes, dtype=np.int32)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)

        return {
            "user_index_batch": user_indexes,
            "impr_index_batch": impr_indexes,
            "clicked_title_batch": click_title_index_batch,
        }

    def load_news_from_file(self, news_file):
        """Read and parse user data from news file.

        Args:
            news_file (str): A file contains several informations of news.

        Returns:
            obj: An iterator that will yields parsed news feature, in the format of dict.
        """
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        cnt = 0

        for index in range(len(self.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(self.news_title_index[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_news_data(
                    news_indexes, candidate_title_indexes,
                )
                news_indexes = []
                candidate_title_indexes = []
                cnt = 0

        if cnt > 0:
            yield self._convert_news_data(
                news_indexes, candidate_title_indexes,
            )

    def _convert_news_data(
            self, news_indexes, candidate_title_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            news_indexes (list): a list of news indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        news_indexes_batch = np.asarray(news_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int32
        )

        return {
            "news_index_batch": news_indexes_batch,
            "candidate_title_batch": candidate_title_index_batch,
        }

    def load_impression_from_file(self, behaivors_file):
        """Read and parse impression data from behaivors file.

        Args:
            behaivors_file (str): A file contains several informations of behaviros.

        Returns:
            obj: An iterator that will yields parsed impression data, in the format of dict.
        """

        if not hasattr(self, "histories"):
            self.init_behaviors(behaivors_file)

        indexes = np.arange(len(self.labels))

        for index in indexes:
            impr_label = np.array(self.labels[index], dtype="int32")
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
                impr_label,
            )

