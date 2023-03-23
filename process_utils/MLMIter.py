#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/22 7:04 PM
# @Author  : taicheng.guo
# @Email:  : 2997347185@qq.com
# @File    : MLMIter.py
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

def load_obj(name, save_root):
    with open(save_root + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class AlignDatset(Dataset):
    def __init__(self, filename, range):
        self.range = range
        self.filename = filename
        if range == 'Model/data':
            if filename == 'MLM_dataset_all_pair':
                self.training_samples = load_obj(filename, "./Alignment/")
                self.title_encodes_en = []
                self.pred_masks_en = []
                self.y = []
                self.title_encodes_no = []
                self.pred_masks_no = []

                for sample in self.training_samples:
                    self.title_encodes_en.append(sample[0])
                    self.pred_masks_en.append(sample[1])
                    self.y.append(sample[2])
                    self.title_encodes_no.append(sample[3])
                    self.pred_masks_no.append(sample[4])

            else:
                self.training_samples = load_obj(filename, "./Alignment/")
                self.title_encodes = []
                self.pred_masks = []
                self.y = []
                for sample in self.training_samples:
                    self.title_encodes.append(sample[0])
                    self.pred_masks.append(sample[1])
                    self.y.append(sample[2])

        elif range == 'Model/engTonor':
            self.training_samples = load_obj(filename, "./Alignment_eng2nor/")
            self.title_encodes_en = []
            self.pred_masks_en = []
            self.y = []
            self.title_encodes_no = []
            self.pred_masks_no = []

            for sample in self.training_samples:
                self.title_encodes_en.append(sample[0])
                self.pred_masks_en.append(sample[1])
                self.y.append(sample[2])
                self.title_encodes_no.append(sample[3])
                self.pred_masks_no.append(sample[4])


    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, index):
        if self.range == 'Model/engTonor':
            return np.array(self.title_encodes_en[index]), \
                   np.array(self.pred_masks_en[index]), \
                   self.y[index], \
                   np.array(self.title_encodes_no[index]), \
                   np.array(self.pred_masks_no[index])

        else:
            if self.filename == 'MLM_dataset_all_pair':
                return np.array(self.title_encodes_en[index]), \
                       np.array(self.pred_masks_en[index]), \
                       self.y[index], \
                       np.array(self.title_encodes_no[index]), \
                       np.array(self.pred_masks_no[index])
            else:
                return np.array(self.title_encodes[index]), \
                       np.array(self.pred_masks[index]), \
                       self.y[index]