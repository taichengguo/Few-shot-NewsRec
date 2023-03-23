import argparse

data_info_n2m = {
	"mind_0_0shot": {
		'user_n': 1000,
		'news_n': 5000,
		'max_title_token': 30,
	},

	"mind_200_4shot":{
		'user_n': 2500,
		'news_n': 8000,
		'max_title_token': 30,
	},

	"adressa": {
		'user_n' : 10000,
		'news_n' : 2097,
		'max_title_token': 30,
	}
}

data_info_m2n = {
	"mind_0_0shot": {
		'user_n': 1000,
		'news_n': 708,
		'max_title_token': 30,
	},

	"mind_200_2shot": {
		'user_n': 1000,
		'news_n': 646,
		'max_title_token': 30,
	},

	"mind_200_4shot":{
		'user_n': 1000,
		'news_n': 708,
		'max_title_token': 30,
	},

	"mind_200_6shot": {
		'user_n': 1000,
		'news_n': 692,
		'max_title_token': 30,
	},

	"mind_5500_10shot": {
		'user_n': 13500,
		'news_n': 1862,
		'max_title_token': 30,
	},

	"adressa": {
		'user_n' : 10000,
		'news_n' : 31099,
		'max_title_token': 30,
	}
}


pretrained_embed = 300
deepwalk_embed = 300
embed_d = 300

def read_args(db='mind', lr=0.0002):

	parser = argparse.ArgumentParser()
	parser.add_argument('--few_shot', type=str, default='_0_0shot')  # ''/'_100'/'_500'/'_1000'
	parser.add_argument('--db', type = str, default = db,
				   help = 'node net dimension')

	parser.add_argument('--embed_d', type = int, default = embed_d,
				   help = 'embedding dimension')

	parser.add_argument('--lr', type = int, default = lr,
				   help = 'learning rate')

	parser.add_argument('--batch_s', type = int, default = 20000,
				   help = 'batch size')

	if db == 'mind':
		mini_batch_s = 80
	else:
		mini_batch_s = 80

	parser.add_argument('--mini_batch_s', type = int, default = mini_batch_s,
				   help = 'mini batch size')

	parser.add_argument('--train_iter_n', type = int, default = 10,
				   help = 'max number of training iteration')
	parser.add_argument("--random_seed", default = 42, type = int)

	parser.add_argument('--save_model_freq', type = float, default = 1,
				   help = 'number of iterations to save model')
	parser.add_argument("--cuda", default = 2, type = int)
	parser.add_argument("--checkpoint", default = '', type=str)
	parser.add_argument("--npratio", default=4, type=int)

	"""
	Some other parameters needed to be test
	"""
	parser.add_argument("--save_emb", default=0, type=int)

	# ablation studies for different modules
	parser.add_argument("--use_PLM", default=1, type=int)
	parser.add_argument("--use_KG", default=0, type=int)
	# few-shot setting
	parser.add_argument("--few_shot_method", default=2, type=int)  # 0-only mind 1-mind+adressa 2-ours
	# other domains
	parser.add_argument("--range", default="Model/engTonor", type=str) # Model/data  Model/engTonor

	parser.add_argument("--align_mode", default="no_freeze", type=str)

	parser.add_argument("--loss_weight", default=0.2, type=float)
	parser.add_argument("--loss_weight_align", default=1, type=float)
	parser.add_argument("--news_cls_iter", default=1, type=int)
	# target domain sim
	parser.add_argument("--target_domain_sim", default=0.6, type=float)

	# top-n plm news
	parser.add_argument("--topn", default=1, type=int)


	args = parser.parse_args()

	if db == 'mind':
		data_key = db + args.few_shot
	else:
		data_key = db

	range = args.range
	if range == 'Model/data' or range == 'Model/new_data' or range != 'Model/engTonor':
		data_info = data_info_n2m
		if data_key in data_info:
			args.A_n = data_info[data_key]['user_n']
			args.P_n = data_info[data_key]['news_n']
			args.max_title_token = data_info[data_key]['max_title_token']
		else:
			args.A_n = data_info["mind_200_4shot"]['user_n']
			args.P_n = data_info["mind_200_4shot"]['news_n']
			args.max_title_token = data_info["mind_200_4shot"]['max_title_token']
	else:
		data_info = data_info_m2n

		if data_key in data_info:
			args.A_n = data_info[data_key]['user_n']
			args.P_n = data_info[data_key]['news_n']
			args.max_title_token = data_info[data_key]['max_title_token']
		else:
			args.A_n = data_info["mind_5500_10shot"]['user_n']
			args.P_n = data_info["mind_5500_10shot"]['news_n']
			args.max_title_token = data_info["mind_5500_10shot"]['max_title_token']

	args.data_path = '../{}/{}/'.format(range, data_key)
	args.model_path = './model_save/{}/'.format(data_key)
	return args



