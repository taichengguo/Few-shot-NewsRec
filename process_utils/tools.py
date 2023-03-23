import torch
import torch.nn as nn
import torch.nn.functional as F
from args import read_args
import numpy as np
import math
args = read_args()
from layers.attention import AdditiveAttention
import random
from tqdm import tqdm
import copy
import pandas as pd

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cross_entropy_loss(c_embed_batch, pos_embed_batch, embed_d):

	batch_size = c_embed_batch.shape[0]
	c_embed = c_embed_batch.unsqueeze(1)  # [B * 1 * emb]

	pos_embed = pos_embed_batch.view(batch_size, -1, embed_d)  # [B * news * emb]
	news_num = pos_embed.shape[1]
	####################################################################################################
	pos_embed = pos_embed.permute(0, 2, 1)
	logits = torch.bmm(c_embed, pos_embed).squeeze(1)
	####################################################################################################
	# cosine similarity
	# c_embed = c_embed.repeat(1, news_num, 1)
	# cos = nn.CosineSimilarity(dim=2, eps=1e-6)
	# logits = cos(c_embed, pos_embed)	# [B * news]
	####################################################################################################

	labels = torch.from_numpy(np.array([0] * batch_size))
	if args.cuda:
		labels = labels.cuda()
	loss = nn.CrossEntropyLoss()(logits, labels)

	# [Batch * 5] [Batch *labels] cross-entropy
	return loss


def align_loss(c_embeddings, sample_embeddings, neg_embeddings, gamma, bz=None):
	"""
	in: center, neg, sample
	out: L
	Returns:
	"""

	if args.align_metric == 'pdist':
		# 2-norm distance
		pdist = nn.PairwiseDistance(p=2)
		c_sample = pdist(c_embeddings, sample_embeddings)
		neg_sample = pdist(neg_embeddings, sample_embeddings)
	elif args.align_metric == 'cos':
		# dot similarity
		cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		c_sample = cos(c_embeddings, sample_embeddings)
		neg_sample = cos(neg_embeddings, sample_embeddings)
	elif args.align_metric == 'dot':
		c_sample = torch.bmm(c_embeddings.view(bz, 1, 300), sample_embeddings.view(bz, 300, 1))
		neg_sample = torch.bmm(neg_embeddings.view(bz, 1, 300), sample_embeddings.view(bz, 300, 1))


	# Ranking loss
	# Hinge = max-margin loss -1=>1 larger value than second input
	target = torch.tensor([1.0] * len(c_embeddings))
	if args.cuda:
		target = target.cuda()

	if args.align_loss == 'rank':
		ranking_func = nn.MarginRankingLoss(margin=float(gamma))
		c_sample = c_sample.float()
		neg_sample = neg_sample.float()
		loss = ranking_func(c_sample, neg_sample, target)
	else:
		# cosine_embedding_loss / MSE loss
		# cosine_distance_func = nn.CosineEmbeddingLoss()
		# loss = cosine_distance_func(c_embeddings, sample_embeddings, target)
		mse_func = nn.MSELoss()
		loss = mse_func(sample_embeddings, c_embeddings)

	return loss


def generate_align_loss(align_start, align_batch, train_samples,
						model_mind, model_ad,
						center, weight=100, gamma=0):

	c_index, samples_index, neg_index = train_samples[0], train_samples[1], train_samples[2]
	align_end = align_start + align_batch

	if align_end >= len(c_index):
		indices = range(align_start, len(c_index))
		align_start = 0
	else:
		indices = range(align_start, align_end)
		align_start = align_start + align_batch

	tem_c_index = [c_index[i] for i in indices]
	tem_neg_index = [neg_index[i] for i in indices]
	tem_sample_index = [samples_index[i] for i in indices]
	node_type = 2

	if center == 'adressa':
		c_embedding = model_ad.model(tem_c_index, node_type, mode='align')
		neg_embedding = model_ad.model(tem_neg_index, node_type, mode='align')
		sample_embedding = model_mind.model(tem_sample_index, node_type, mode='align')
	else:
		c_embedding = model_mind.model(tem_c_index, node_type, mode='align')
		neg_embedding = model_mind.model(tem_neg_index, node_type, mode='align')
		sample_embedding = model_ad.model(tem_sample_index, node_type, mode='align')

	# TODO 固定住Source domain的embedding
	c_embedding = c_embedding.detach()
	neg_embedding = neg_embedding.detach()
	# transfer embeddings of two domains
	aligned_loss = weight * align_loss(c_embedding, sample_embedding, neg_embedding, gamma=gamma, bz=len(indices))
	return aligned_loss, align_start

