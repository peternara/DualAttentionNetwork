# coding=utf-8
# main script for memory qa training and testing



d = "giving the preprocessed data ,train or test model"

import sys,os,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # so here won't have poll allocator info

import cPickle as pickle
import numpy as np

#from model import get_model # S1...S7, then Smm attention, approximate the attention tensor
#from model_v2 import get_model # using the attention cube
from trainer import Trainer
from tester import Tester
import math,time,json,random

import tensorflow as tf

from tqdm import tqdm

from utils import Dataset,update_config,evaluate,getEvalScore,sec2time

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

get_model = None # the model we will use, based on parameter in the get_args()

def get_args():
	global get_model
	parser = argparse.ArgumentParser(description=d)
	parser.add_argument("prepropath",type=str)
	parser.add_argument("outbasepath",type=str,help="full path will be outbasepath/modelname/runId")
	parser.add_argument("modelname",type=str,default="dualatt")
	parser.add_argument("--runId",type=int,default=0,help="used for run the same model multiple times")


	parser.add_argument("--featpath",default=None,help="need this if there is no feature npz")
	parser.add_argument("--feat_dim",default=None,help="14,14,2048")
	
	parser.add_argument("--load",action="store_true",default=False,help="whether to load existing model")
	parser.add_argument("--load_best",action="store_true",default=False,help="whether to load the best model")
	# use for pre-trained model
	parser.add_argument("--load_from",type=str,default=None)
	parser.add_argument("--ignore_vars",type=str,default=None)

	parser.add_argument("--is_train",action="store_true",default=False,help="training mode, ")
	parser.add_argument("--is_test",action="store_true",default=False,help="testing mode, otherwise test mode")

	parser.add_argument("--save_answers",action="store_true",default=False,help="save testing img2sent sent2img answer")


	parser.add_argument("--is_test_on_val",action="store_true",default=False,help="test on validation set")
	parser.add_argument("--is_save_weights",action="store_true",default=False,help="whether to save model weights to val_path")
	parser.add_argument("--is_save_vis",action="store_true",default=False,help="whether to save each layer output for visualization during testing, will save into val_path")


	parser.add_argument("--save_period",type=int,default=200,help="num steps to save model and eval")
	parser.add_argument("--val_path",type=str,default="",help="path to store the eval file[for testing]")

	parser.add_argument("--hn_num",type=int,default=16,help="hard negative mining num")

	parser.add_argument("--is_pack_model",action="store_true",default=False,help="with is_test, this will pack the model to a path instead of testing")
	parser.add_argument("--pack_model_path",type=str,default=None,help="path to save model")
	parser.add_argument("--pack_model_note",type=str,default=None,help="leave a note for this packed model for future reference")

	# ---------------------------- training hparam

	#training detail
	parser.add_argument('--batch_size',type=int,default=20)
	# no val_num_batches, full val will be used
	#parser.add_argument('--val_num_batches',type=int,default=100,help="eval during training, get how many batch in train/val to eval")

	parser.add_argument("--num_epochs",type=int,default=20) # num_step will be num_example/batch_size * epoch
	# drop out rate
	parser.add_argument('--keep_prob',default=1.0,type=float,help="1.0 - drop out rate;remember to set it to 1.0 in eval")

	# l2 weight decay rate
	parser.add_argument("--wd",default=None,type=float,help="l2 weight decay loss, 0.002 is a good number, default not applied")

	parser.add_argument("--clip_gradient_norm",default=None,type=float,help="gradient will be in (-1*clip_gradient_norm , 1* clip_gradient_norm), try 0.1")
	parser.add_argument("--optimizer",default="momentum",help="momentum|adadelta|adam")

	parser.add_argument("--learning_rate_decay",default=0.95,type=float,help=("learning rate decay"))
	parser.add_argument("--learning_rate_decay_examples",default=500000,type=int,help=("how many sample to have one decay"))

	parser.add_argument("--init_lr",default=0.5,type=float,help=("Start learning rate"))

	#------------------------------------------ all kinds of threshold

	# cap of the word
	parser.add_argument('--word_count_thres',default=2,type=int,help="word count threshold")
	parser.add_argument('--char_count_thres',default=10,type=int,help="char count threshold")

	# sentence length
	# train max 82, val max 56, test max 49
	parser.add_argument('--sent_size_thres',default=100,type=int,help="max sentence word count for album_title")

	parser.add_argument('--word_size_thres',default=16,type=int,help="max word character count")

	#------------------------------------------------------------------------------
	# -------model detail
	parser.add_argument("--num_hops",type=int,default=2,help="memory network hop times")

	#parser.add_argument("--mcb",action="store_true", help="MCB attention model exp")
	#parser.add_argument("--mcb_outdim",type=int,default=16000, help="MCB attention model exp")

	parser.add_argument('--margin',default=100,type=float,help="margin of the rank loss")


	# model detail
	parser.add_argument('--hidden_size',type=int,default=512,help="hidden size for rnn")

	parser.add_argument("--concat_rnn",action="store_true",default=False,help="concat bidirectional rnn output")

	# for calculationg T_att, use concat instead of the dual att paper
	parser.add_argument("--concat_att",action="store_true",default=False,help="concat to get T_att")


	# whether to finetune word2vec
	parser.add_argument("--finetune_wordvec",default=False,action="store_true",help="finetuning the pre-trained wordvec")
	# not using pre-trained word2vec
	parser.add_argument("--no_wordvec",default=False,action="store_true",help="not to use pre-train word vectors")
	parser.add_argument("--word_emb_size",default=0,type=int,help="need to be set when no using pre train vectors")

	# whether to use char emb
	parser.add_argument("--use_char",default=False,action="store_true",help="use character CNN embeding")
	# char embeding size
	parser.add_argument('--char_emb_size',default=8,type=int,help="char-CNN channel size")
	parser.add_argument("--char_out_size",default=100,type=int,help="char-CNN output size for each word")
	

	parser.add_argument("--batch_norm",default=False,action="store_true",help="apply batch norm before linear functino 's non-linearity")

	# useless, dont use
	# or Residual network
	#parser.add_argument("--use_residual",default=False,action="store_true",help="use residual network")
	#parser.add_argument("--residual_layer_num",default=2,type=int,help=("residual network for word and char"))

	args = parser.parse_args()

	if args.no_wordvec:
		assert args.word_emb_size > 0

	if args.is_pack_model:
		assert args.is_test,"use pack model with is_test"
		assert args.pack_model_path is not None, "please provide where pack model to"
		assert args.pack_model_note is not None, "please provide some note for the packed model"


	from model import get_model 

	args.outpath = os.path.join(args.outbasepath,args.modelname,str(args.runId).zfill(2))
	mkdir(args.outpath)

	args.save_dir = os.path.join(args.outpath, "save")#,"save" # tf saver will be save/save-*.meta
	mkdir(args.save_dir)
	args.save_dir_model = os.path.join(args.save_dir,"save") # tf saver will be save/save-*step*.meta

	args.save_dir_best = os.path.join(args.outpath, "best")
	mkdir(args.save_dir_best)
	args.save_dir_best_model = os.path.join(args.save_dir_best,"save-best")

	args.write_self_sum = True
	args.self_summary_path = os.path.join(args.outpath,"train_sum.txt")

	args.record_val_perf = True
	args.val_perf_path = os.path.join(args.outpath,"val_perf.p")

	
	if args.load_best:
		args.load = True

	# no need, we squash to 2d
	#if args.use_bidirection:
	#	assert args.use_question_att, "need all matrix use bidirectional attention to match the dimension"
	#	assert args.use_choices_att, "need all matrix use bidirectional attention to match the dimension"

	# if test, has to load
	if not args.is_train:
		assert args.is_test, "if not train, please use is_test flag"
		args.load = True
		args.num_epochs = 1
		args.keep_prob = 1.0
		#assert args.val_path!="","Please provide val_path"
		if args.val_path == "":
			if args.load_best:
				args.val_path = os.path.join(args.outpath,"test_best")
			else:
				args.val_path = os.path.join(args.outpath,"test")
		print "test result will be in %s"% args.val_path
		mkdir(args.val_path)

		args.vis_path = os.path.join(args.val_path,"vis")
		args.weights_path = os.path.join(args.val_path,"weights")
		if args.is_save_vis:
			mkdir(args.vis_path)
			print "visualization output will be in  %s"% args.vis_path
		if args.is_save_weights:
			mkdir(args.weights_path)
			print "model weights will be in %s"% args.weights_path

	return args



def read_data(config,datatype,loadExistModelShared=False,subset=False): 
	data_path   = os.path.join(config.prepropath,"%s_data.p"%datatype) # prepro/train_data.p, prepro/val_data.p
	shared_path = os.path.join(config.prepropath,"%s_shared.p"%datatype) # prepro/train_shared.p, prepro/val_shared.p
	
	with open(data_path,"rb")as f: # prepro/train_data.p, prepro/val_data.p
		data = pickle.load(f)
	# 아래 부분에 좀더 자세한 내용이 있음.	
	# data = (imageid, sentence_word[], sentence_word_char[[]])
	# 1 set 예 -  
	#	('442983801', 
	#	['a', 'very', 'young', 'girl', 'is', 'holding', 'food', 'while', 'sitting', 'in', 'a', 'highchair', '.'],
	#	[['a'], ['v', 'e', 'r', 'y'], ['y', 'o', 'u', 'n', 'g'], ['g', 'i', 'r', 'l'], ['i', 's'], ['h', 'o', 'l', 'd', 'i', 'n', 'g'], ['f', 'o', 'o', 'd'], ['w', 'h', 'i', 'l', 'e'], ['s', 'i', 't', 't', 'i', 'n', 'g'], ['i', 'n'], ['a'], ['h', 'i', 'g', 'h', 'c', 'h', 'a', 'i', 'r'], ['.']]),
		
	with open(shared_path,"rb") as f: # prepro/train_shared.p, prepro/val_shared.p
		shared = pickle.load(f) # this will be added later with word id, either new or load from exists
	# 아래 부분에 좀더 자세한 내용이 있음.	
	# shared - (word:1)	
	# 예 - 
	#	... 'groomer': 1, 'ywca': 1, 'footwork': 1, 'shane': 1, 'tinged': 1, 'jumpos': 1, 'wight': 1, 'gondolas': 1, '5028': 1, 'wavelength': 1, '66197': 1, 'bazzar': 1, 'apportioned': 1, 'parasailors': 1, 'rotting': 1, 'jewel': 1, 'emery': 1, 'intentionally': 1, 'pods': 1})}
	# 현상태는 존재하는 word:1로 셋팅한 파일정보인듯

	# load the imgid2feat from separate npz file
	shared['imgid2feat'] = None
	shared['featpath'] = None

	# config.featpath = resnet-152/, 미리 뽑은 image feature 디렉토리,  3217909454.npy
	if config.featpath is None:
		imgid2featpath = os.path.join(config.prepropath,"%s_imgid2feat.npz"%datatype)
		shared['imgid2feat'] = dict(np.load(imgid2featpath))
	else:
		#  위 말고 여기를 통과
		assert config.feat_dim is not None # config.feat_dim : [14, 14, 2048] 
		if type(config.feat_dim) == type("a"): # 여길 통과하지 않음. 다만 string 값으로 주어지는듯
			config.feat_dim = [int(one) for one in config.feat_dim.split(",")] 
		
		shared['featpath'] = config.featpath # config.featpath = resnet-152(directory)
		#shared['featCache'] = {}
		#shared['cacheSize'] = 0 # one for train and one for val
       
        # data['data']의 구성은 (이미지 아이디, sentence의 word [], sentence word들의 character[[],])
	# example) 1 unit > ('240696675', ['dog', 'running', 'on', 'narrow', 'dirt', 'path', '.'], [['d', 'o', 'g'], ['r', 'u', 'n', 'n', 'i', 'n', 'g'], ['o', 'n'], ['n', 'a', 'r', 'r', 'o', 'w'], ['d', 'i', 'r', 't'], ['p', 'a', 't', 'h'], ['.']]), 
	num_examples = len(data['data']) # (imageId,sent,sentc) 
	# num_examples = 148915(=학습 전체 이미지 29783개 x Caption 5, trainingset), 5000(valset)
	# 	이 파일정보에서 얻을수 있음 : ../../../DeViSE/DB/flickr30k_parse/flickr30k_results_20130124.token
	#		예) flickr30k_results_20130124.token
	#		1000092795.jpg#0        Two young guys with shaggy hair look at their hands while hanging out in the yard .
	#		1000092795.jpg#1        Two young , White males are outside near many bushes .
	#		1000092795.jpg#2        Two men in green shirts are standing in a yard .
	#		1000092795.jpg#3        A man in a blue shirt standing in a garden .
	#		1000092795.jpg#4        Two friends enjoy time spent together .
 	# 궁금한점은 이미지 id는 동일한데..5개씩가지고 있으면, 중복될텐데..문제가 발생할수 있는데 어떻게 구성될까?
	# > 일단 기본 format으로 재구성하는 prepro_flickr30k.py을 봐야함.
	# prepro_flickr30k.py 은
	#	1. imagepath, sentence, imageid, sentnceid로 구분
	#		- 예) 1000092795.jpg#0        Two young guys with shaggy hair look at their hands while hanging out in the yard .
	#			imageId - 1000092795
	#			sentid  - 1000092795.jpg#0 
	#			text    - Two young guys with shaggy hair look at their hands while hanging out in the yard .
	#			그리고 이런형태로 먼저 저장 > imgid2text[imageId].append((sentid,text.strip()))
	#	2. image id와 sentence를 가져와서, 다음과 같이 imageid, worde단위, char단위로 구분한다.	
	# 		4741726894
	#		['a', 'woman', 'wearing', 'a', 'green', 'shirt', 'is', 'sitting', 'on', 'a', 'bench', 'in', 'front', 'of', 'two', 'elephant', 'statues', '.']
	#		[['a'], ['w', 'o', 'm', 'a', 'n'], ['w', 'e', 'a', 'r', 'i', 'n', 'g'], ['a'], ['g', 'r', 'e', 'e', 'n'], ['s', 'h', 'i', 'r', 't'], ['i', 's'], ['s', 'i', 't', 't', 'i', 'n', 'g'], ['o', 'n'], ['a'], ['b', 'e', 'n', 'c', 'h'], ['i', 'n'], ['f', 'r', 'o', 'n', 't'], ['o', 'f'], ['t', 'w', 'o'], ['e', 'l', 'e', 'p', 'h', 'a', 'n', 't'], ['s', 't', 'a', 't', 'u', 'e', 's'], ['.']]
	#		그리고 이런형태로 저장 >  
	#			data.append((imgid, si, csi)) # (imageid, worde단위, char단위)
	#			sentid2data[sentid] = {"sents":si,"sents_c":csi}
	#	3.save - d와 share의 두가지 형태로 두파일로 저장
	#		d = {
	#                'data':data, # (imgid,si,csi)
	#                "sentids":sentids, # (sentid)
	#                "imgids":imgids,
 	#               #"sentid2data":sentid2data
 	#               }	
	#		> train_data.p - 요런 형태로 저장 - {'imgids': ['196521598', '19663315',..],   'data': [('196521598', ['a',,,], [].., 'sentids': ['196521598.jpg#0', ... '295729735.jpg#4']}         
	#
	#		shared = {
 	#                       #"imgid2feat":imgid2feat, # save the image feature to a separete npz file
   	#                       "word_counter":word_counter,
 	#                       "char_counter":char_counter,
  	#                      "word2vec":{}, # or "word2vec":word2vec,
 	#               }
	#               > train_shared.p - 요런 형태로 저장 - {'char_counter': Counter({'a': 28373, 'e': 23357, 'n': 22336, 'i': 20714, 'o': 18937,,,}, 'word2vec': {}, 'word_counter': Counter({'a': 8768, '.': 4766, 'in': 2633,,,,'pods': 1})}
		
	
	"""
	if(filter_data): # TODO: no filter implemented
		masks = []
		for i in xrange(num_examples):
			masks.append(data_filter(data,i,config,shared))
		valid_idxs = [i for i in xrange(len(masks)) if masks[i]]
	"""
	
	# trainset = 148915 이니 [0,,,,148914]
	# valset = 5000 이니 [0,,,,4999]
	valid_idxs = range(num_examples)
        
	# loaded 148915/148915 data points for train
        # loaded 5000/5000 data points for val
	print "loaded %s/%s data points for %s"%(len(valid_idxs),num_examples,datatype)

	# 다음과 같이 생성되는 파일 및 디렉토리
	#       ls models/dan/00/
        #       best  save  shared.p
	# this is the file for the model' training, with word ID and stuff, if set load in config, will read from existing, otherwise write a new one
	model_shared_path = os.path.join(config.outpath,"shared.p") # note here the shared.p is for one specific model
        # model_shared_path = models/dan/00/shared.p			
	if(loadExistModelShared): # tain is loadExistModelShared=False, not train > val set 
		# models/dan/00/shared.p	
		with open(model_shared_path,"rb") as f:
			model_shared = pickle.load(f)
		# word2idx, char2idx 정보가 있다.
		# 실제 예
		# ... 'sloping': 11790, 'album': 11792, 'junk': 11793, 'kinds': 5914, 'shredded': 5915, 'mulch': 11794, 'pumps': 5916, 'upwards': 11795, 'barista': 10647, 'ranks': 5918, 'gelati': 11796, 'dominoes': 5919, 'sleeves': 6909, 'laundromat': 5920, 
		# 'sash': 5921}, 'char2idx': {'<UNK>': 1, '!': 2, '#': 3, '%': 4, '$': 5, '&': 6, ',': 7, '.': 8, '1': 9, '0': 10, '3': 11, '2': 12, '5': 13, '4': 14, '7': 15, '6': 16, '9': 17, '8': 18, ';': 19, '?': 20, '<NULL>': 0, 'a': 22, 'c': 23, 'b': 21, 'e': 24, 'd': 25, 'g': 26, 'f': 27, 'i': 28, 'h': 29, 'k': 30, 'j': 31, 'm': 32, 'l': 33, 'o': 34, 'n': 35, 'q': 36, 'p': 37, 's': 38, 'r': 39, 'u': 40, 't': 41, 'w': 42, 'v': 43, 'y': 44, 'x': 45, 'z': 46}}
			
		for key in model_shared:
			shared[key] = model_shared[key]
			# key = word2idx
			# model_shared[key] = {'raining': 2, 'writings': 5922, 'childern': 5923, 'foul': 3, 'revelers': 19,....
			# key = char2idx
			# model_shared[key] = {'<UNK>': 1, '!': 2, '#': 3, '%': 4, '$': 5, '&': 6, ',': 7, '.': 8, '1': 9, '0': 10, '3': 11, '2': 12, '5': 13, '4': 14, '7': 15, '6': 16, '9': 17, '8': 18, ';': 19, '?': 20, '<NULL>': 0, 'a': 22, 'c': 23, 'b': 21, 'e': 24, 'd': 25, 'g': 26, 'f': 27, 'i': 28, 'h': 29, 'k': 30, 'j': 31, 'm': 32, 'l': 33, 'o': 34, 'n': 35, 'q': 36, 'p': 37, 's': 38, 'r': 39, 'u': 40, 't': 41, 'w': 42, 'v': 43, 'y': 44, 'x': 45, 'z': 46}
		# no fine tuning of word vector
		if config.no_wordvec:
			shared['word2vec'] = {}
	else:
		# train - train 과정에서 run - 생성
		# no fine tuning of word vector
		if config.no_wordvec:
			shared['word2vec'] = {}
		
		if config.finetune_wordvec:
			# here to finetune, so the word in the word2vec
			shared['word2idx'] = {word:idx+2 for idx,word in enumerate([word for word,count in shared['word_counter'].items() if (count > config.word_count_thres) or shared['word2vec'].has_key(word)])}

		else:
			# 여길 진행 - word2vec가 아닌 실제로 raw 레벨(one hot)에서 embedding을 학습하는것같음
			# the word larger than word_count_thres and not in the glove word2vec
			# word2idx -> the idx is the wordCounter's item() idx 
			# the new word to index
			# 
			# 일률적으로 단어의 id를 매긴다
			shared['word2idx'] = {word:idx+2 for idx,word in enumerate([word for word,count in shared['word_counter'].items() if (count > config.word_count_thres) and not shared['word2vec'].has_key(word)])}
			'''
			# 위와 같다
			word_list = []
               		for word, count in shared['word_counter'].items():
                        	#print word, count # ex- hilfiger 2
                        	if (count > config.word_count_thres):
                                	word_list.append(word)

                	shared['word2idx'] = {}
                	for idx,word in enumerate(word_list):
                        	#print idx, word
                        	shared['word2idx'].update({word:idx+2})
                	print shared['word2idx']
                	print len(shared['word2idx'])
			'''
			# 2번부터 시작하는듯~ 
                        # shared['word2idx'] : word to index
			# example) {.....,'smilling': 11667, 'walk': 5774, 'packaging': 11668,....}

		shared['char2idx'] = {char:idx+2 for idx,char in enumerate([char for char,count in shared['char_counter'].items() if count > config.char_count_thres])}
		# char to index
		# example : {'!': 2, '#': 3, '%': 4, '$': 5, '&': 6, ',': 7, '.': 8, '1': 9, '0': 10, '3': 11, '2': 12, '5': 13, '4': 14, '7': 15, '6': 16, '9': 17, '8': 18, ';': 19, '?': 20, 'a': 22, 'c': 23, 'b': 21, 'e': 24, 'd': 25, 'g': 26, 'f': 27, 'i': 28, 'h': 29, 'k': 30, 'j': 31, 'm': 32, 'l': 33, 'o': 34, 'n': 35, 'q': 36, 'p': 37, 's': 38, 'r': 39, 'u': 40, 't': 41, 'w': 42, 'v': 43, 'y': 44, 'x': 45, 'z': 46}
		
		# print "len of shared['word2idx']:%s"%len(shared['word2idx']) 
      		# len of shared['word2idx']:11796
		# 학습셋에 사용되는 word의 수
		
		# NULL, 시작
		NULL = "<NULL>"
		UNK = "<UNK>" 
		shared['word2idx'][NULL] = 0
		shared['char2idx'][NULL] = 0
		shared['word2idx'][UNK] = 1
		shared['char2idx'][UNK] = 1

		# model_shared_path = models/dan/00/shared.p 의 이름으로 저장
		#    shared['word2idx'] & shared['char2idx'] 순으로 데이터를 저장
		#    see models/dan/00/shared.p 파일을 보면 알수 있다.
'''
(dp1
S'word2idx'
p2
(dp3
S'raining'
p4
I2
sS'writings'
p5
I5922
sS'childern'
p6
I5923
sS'both'
p7
...
sS'sash'
p11760
I5921
ssS'char2idx'
p11761
(dp11762
g638
I1
sS'!'
I2
sS'#'
I3
sS'%'
I4
sS'$'
I5

'''
		# existing word in word2vec will be put after len(new word)+2(0:<NULL>, 1:<UNK>)
		# model_shared_path = models/dan/00/shared.p
		pickle.dump({"word2idx":shared['word2idx'],'char2idx':shared['char2idx']},open(model_shared_path,"wb"))


	# load the word embedding for word in word2vec

	# word2idx is not in the word2vec if not finetune
	# existing_word2idx is the word in word2vec not in word2idx
	# word in word2vec -> idx , idx is the word2vec items() 's idx
	# this will change for the same word in different dataset split

	# this could be empty if finetune
	shared['existing_word2idx'] = {word:idx for idx,word in enumerate([word for word in sorted(shared['word2vec'].keys()) if not shared['word2idx'].has_key(word)])}
	# shared['existing_word2idx'] = {} # train, val
	
	# idx -> vector
	# this could be empty if finetune
	idx2vec = {idx:shared['word2vec'][word] for word,idx in shared['existing_word2idx'].items()}
	# idx2vec = {} # train, val
	# load all this vector into a matrix
	# so you can use word -> idx -> vector
	# using xrange(len) so that the idx is 0,1,2,3...
	# then it could be call with embedding lookup with the correct idx

	# could be empty
	shared['existing_emb_mat'] = np.array([idx2vec[idx] for idx in xrange(len(idx2vec))],dtype="float32")
	# shared['existing_emb_mat'] = [] # train, val
	print "shared['existing_emb_mat'].shape:%s"%(list(shared['existing_emb_mat'].shape))

	# get the image feature into one matrix as well
	# we have to load img feature on the fly since train val has different image feature
	"""
	shared['imgid2idx'] = {imgid:idx for idx,imgid in enumerate([imgid for imgid in sorted(shared['imgid2feat'].keys())])}
	imgidx2vec = {idx:shared['imgid2feat'][imgid] for imgid,idx in shared['imgid2idx'].items()}
	shared['img_feat'] = np.array([imgidx2vec[idx] for idx in xrange(len(imgidx2vec))],dtype="float32")
	"""


	return Dataset(data,datatype,shared=shared,is_train=datatype=="train",imgfeat_dim=config.feat_dim)

def load_feats(imgid2idx,shared,config):
	image_feats = np.zeros([len(imgid2idx)] + config.feat_dim,dtype="float32")
	for imgid in imgid2idx: # fill each idx with feature, -> pid

		if shared['imgid2feat'] is None: # shared['imgid2feat'] = None, file type load
			"""
			# load feature online
			assert shared['featpath'] is not None
			if shared['featCache'].has_key(imgid):
				feat = shared['featCache'][imgid]
			else:
				feat = np.load(os.path.join(shared['featpath'],"%s.npy"%imgid))
				if len(shared['featCache']) <= shared['cacheSize']:
					shared['featCache'][imgid] = feat
			"""
			# local conv featue file path = os.path.join(shared['featpath'],"%s.npy"%imgid)
			# 	ex) resnet-152/523488750.npy
			feat = np.load(os.path.join(shared['featpath'],"%s.npy"%imgid)) # (14, 14, 2048)
		else:
			feat = shared['imgid2feat'][imgid]
		image_feats[imgid2idx[imgid]] = feat
	return image_feats

def train(config):
	self_summary_strs = [] # summary string to print out for later
	val_perf          = [] # summary of validation performance

	############################################################################################
	# flickr30k data
	# 	image : 31783 인데 29783개의  학습셋
	#	trainset : mRNN/flickr30K_train_list_mRNN.txt :  29783개
	#       valset : mRNN/flickr30K_val_list_mRNN.txt  : 1000
	#	testset : mRNN/flickr30K_test_list_mRNN.txt : 1000
	############################################################################################
	
	# first, read both data and filter stuff,  to get the word2vec idx,
	train_data         = read_data(config, 'train', config.load) # config.load = False
	config.imgfeat_dim = train_data.imgfeat_dim
	# print train_data.imgfeat_dim = train
	
	val_data           = read_data(config, 'val', True, subset=False) # dev should always load model shared data(word2idx etc.) from train		
	
	# now that the dataset is loaded , we get the max_word_size from the dataset
	# then adjust the max based on the threshold as well
	# also get the vocab size
	config_vars = vars(config)
	str_ = "threshold setting--\n" + "\t"+ " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.thresmeta])
	# threshold setting--
        # sent_size_thres:200 ,word_size_thres:20
	print str_		
	self_summary_strs.append(str_)

	# cap the numbers
	# max sentence word count etc.
	update_config(config,[train_data,val_data],showMeta=True) # all word num is <= max_thres   

	str_ = "renewed ----\n"+"\t" + " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.maxmeta])
	# renewed ----
        # max_sent_size:82 ,max_word_size:16
	print str_
	self_summary_strs.append(str_)


	# now we initialize the matrix for word embedding for word not in glove
	word2vec_dict = train_data.shared['word2vec'] # empty if not use pre-train vector
	# empty
	# 실제값은 {}
	
	word2idx_dict = train_data.shared['word2idx'] # this is the word not in word2vec # for finetuning or not using word2vec then it is all the word 
	# word to index 
	# 실제값은  {...  'dominoes': 5919, 'sleeves': 6909, 'laundromat': 5920, 'sash': 5921}
	
	# we are not fine tuning , so this should be empty; empty if not use pretrain vector
	# if finetuning , this will have glove
	idx2vec_dict = {word2idx_dict[word]:vec for word,vec in word2vec_dict.items() if word in word2idx_dict}
	# idx2vec_dict is empty
	# 실제값은 {} 
	
	# len(idx2vec_dict):0,word_vocab_size:11798
	print "len(idx2vec_dict):%s,word_vocab_size:%s"%(len(idx2vec_dict),config.word_vocab_size)

	# config.word_vocab_size = len(train_data.shared['word2idx']) # the word not in word2vec
	# so the emb_mat should all be a random vector
	# np.random.multivariate_normal gets mean of zero and co of 1 for each dim, like 
	#>>> np.random.multivariate_normal(np.zeros(5),np.eye(5))
	#array([-0.73663652, -1.16417783, -0.74083293, -0.80016731,  0.060182  ])

	# random initial embedding matrix for new words
	# this will take a long time when vocab_size is 6k
	if not config.no_wordvec: # config.no_wordvec is True : 따라서, 여긴 진행하진 않는다.
		config.emb_mat = np.array([idx2vec_dict[idx] if idx2vec_dict.has_key(idx) else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size)) for idx in xrange(config.word_vocab_size)],dtype="float32") 
		# for idx in xrange(config.word_vocab_size):
                #	if idx2vec_dict.has_key(idx)
                #	else: np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                #	print idx, idx2vec_dict.has_key(idx) # idx:11567, idx2vec_dict.has_key(idx)
		#		, False 이면 밑의 random 초기화 변수를 선택한다.
                #	print np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                #	print idx2vec_dict[idx]	

	# from model import get_model	
	model = get_model(config) # construct model under gpu0

	#for var in tf.trainable_variables():
	#	print var.name,var.get_shape()
	#sys.exit()

	trainer = Trainer(model,config)
	tester = Tester(model,config)
	saver = tf.train.Saver(max_to_keep=5) # how many model to keep
	bestsaver = tf.train.Saver(max_to_keep=5) # just for saving the best model

	save_period = config.save_period # also the eval period

	# start training!
	# allow_soft_placement :  tf will auto select other device if the tf.device(*) not available
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session(config=tfconfig) as sess:

		# calculate total parameters 
		totalParam = cal_total_param()
		str_ = "total parameters: %s"%(totalParam)
		print str_
		self_summary_strs.append(str_)
		
		# config.load = False
		# config.load_best = False		
		initialize(load=config.load,load_best=config.load_best,model=model,config=config,sess=sess)

		# the total step (iteration) the model will run
		last_time = time.time()
		# total / batchSize  * epoch
		num_steps = int(math.ceil(train_data.num_examples/float(config.batch_size)))*config.num_epochs
		# num_steps = 34920
		# train_data.num_examples = 148915
		# config.batch_size       = 256
		# config.num_epochs       = 60
		
		# get_batches is a generator, run on the fly
		# there will be num_steps batch
		str_ = " batch_size:%s, epoch:%s, %s step every epoch, total step:%s,eval/save every %s steps"%(config.batch_size,config.num_epochs,math.ceil(train_data.num_examples/float(config.batch_size)),num_steps,config.save_period)
		print str_
		#  str_ = batch_size:256, epoch:60, 582.0 step every epoch, total step:34920,eval/save every 1000 steps
		
		self_summary_strs.append(str_)

		metric = "t2i_r@5" # TODO: change this?
		best = {metric:0.0,"step":-1} # remember the best eval acc during training

		finalperf = None
		isStart = True

		for batch in tqdm(train_data.get_batches(config.batch_size,no_img_feat=True,num_steps=num_steps),total=num_steps,ascii=True):
			# each batch has (batch_idxs,Dataset(batch_data, full_shared))
			# batch_data has {"q":,"y":..."pidx2feat",.."photo_idxs"..}

			global_step = sess.run(model.global_step) + 1 # start from 0

			# if load from existing model, save if first
			if (global_step % save_period == 0) or (config.load_best and isStart) or (config.load and isStart and (config.ignore_vars is None)):

				isStart=False

				tqdm.write("\tsaving model %s..."%global_step)
				saver.save(sess,config.save_dir_model,global_step=global_step)
				tqdm.write("\tdone")

				evalperf = evaluate(val_data,config,sess,tester)

				tqdm.write("\teval on validation:%s, (best %s:%s at step %s) "%(evalperf,metric,best[metric],best['step']))

				# remember the best acc
				if(evalperf[metric] > best[metric]):
					best[metric] = evalperf[metric]
					best['step'] = global_step
					# save the best model
					tqdm.write("\t saving best model...")
					bestsaver.save(sess,config.save_dir_best_model,global_step=global_step)
					tqdm.write("\t done.")

				finalperf = evalperf
				val_perf.append(evalperf)
				
			# 여기는 모든 데이터 정보로부터 batch단위로 재구성한다.
			batchIdx, batch_data = batch
			batch_data = batch_data.data
			# each batch is ['pos'],['neg']
			# hard negative mining
			# pair each pos image with max_similarity(pos_img,neg_sent)
			# and pos sent with max_similarity(pos_sent,neg_img)
			assert len(batch_data['pos']) == len(batch_data['neg'])
			# TODO: select hn_num of negative here, save computation?
			alldata = batch_data['pos'] + batch_data['neg'] #  (imgid,sent,sent_c)					
			#  1. get all pos and neg's image and sentence embeddding			
			#for one in alldata:
                        #       print one[0] # image id
                        #       print one[1] # sentence_word
                        #       print one[2] # sentence_word_char
			#
			#  ??? all_imgs의 길이는 512(156+256)이어야하는데, 이보다 조금 못미칠경우가 있다.
			#  	print len(alldata) = 512, 인데,  	
			#  	print len(all_imgs) = 508 ??
			# 512, alldata 안에 확인결과 중복이 존재할수도 있다.
			#	이 중복은 positive data에서 발생, 이를 제외한 negative data를 만들었으니 당연~
			#	그래서, 좀 바꿔야하지 않을까함	
			
			all_imgs  = list(set([one[0] for one in alldata])) # so, batch image ids
			imgid2idx = {}		
			
			for imgid in all_imgs: # batch image ids = 508이라면
				imgid2idx[imgid] = len(imgid2idx.keys())
				# print imgid, imgid2idx[imgid]
				# 	2881441125 0
				#       ...
				# 	51146082 472
				# 	2198484810 473
				#	769934076 474
				# 	2874876837 475
				#       ...
				#	3681637675 507
				
			# load the actual image feature matrix
			# 	conv image feature file load
			image_feats = load_feats(imgid2idx,train_data.shared,config)
			
			# only 학습셋이 아니라 테스트셋을 위한 재구축을 위한 batch
			# 	: alldata = batch_data['pos'] + batch_data['neg'] 
			mining_batch                = {}
			mining_batch['imgs']        = [one[0] for one in alldata] # image ids
			mining_batch['imgid2idx']   = imgid2idx
			mining_batch['imgidx2feat'] = image_feats
			
			# image id를  제외시킨 데이터 = a list of (sentence_word, sentence_word_char)
			mining_batch['data']        = [(one[1],one[2]) for one in alldata]
			
			# harg negative sample를 구하기 위한 작업
			# mining_batch, N_pos+N_neg
			z_u, z_v = tester.step(sess,(batchIdx,Dataset(mining_batch, "test", shared=train_data.shared, is_train=False, imgfeat_dim=config.feat_dim)))
			# z_u : the output of the embedding for text, (512, 3, 512)
			#	2번째 dimension(=3)의 의미가 모지??
			# z_v : the output of the embedding for images, (512, 3, 512)		
			
			assert len(z_u) == len(z_v),(len(z_u),len(z_v))
			z_u_pos = z_u[:len(batch_data['pos'])]
			z_v_pos = z_v[:len(batch_data['pos'])]
			z_u_neg = z_u[len(batch_data['pos']):]
			z_v_neg = z_v[len(batch_data['pos']):]
			assert len(z_u_pos) == len(z_v_pos) == len(z_u_neg) == len(z_v_neg) == len(batch_data['pos']),(len(z_u_pos),len(z_v_pos),len(z_u_neg),len(z_v_neg),len(batch_data['pos']))
			
			# 2. ensemble (pos_img,pos_sent,neg_img,neg_sent) batch
			# 2.1 for each pos_img ,find the best neg_sent,
			posimg2negsentIdxs = {} # pos idx :0 -> N-1, ====> neg idx :0 ->N-1 
			possent2negimgIdxs = {}

			check_num = config.hn_num # 32
			neg_idxs  = range(len(z_u_neg)) # 256
			# neg_idxs > [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
			
			# harg negative sample를 구하기 위한 작업
			for i in xrange(len(batch_data['pos'])):				
				pos_img_vec = z_v_pos[i] # [hop+1,d]
				pos_sent_vec = z_v_pos[i] # [hop+1,d]
				check_neg_idxs = random.sample(neg_idxs,check_num)
				simis = np.zeros((2,check_num),dtype="float")
				for j,check_neg_idx in enumerate(check_neg_idxs):
					neg_sent_vec = z_u_neg[check_neg_idx]
					neg_img_vec = z_v_neg[check_neg_idx]

					simis[0,j] = np.sum(pos_img_vec*neg_sent_vec)
					simis[1,j] = np.sum(pos_sent_vec*neg_img_vec)

				posimg2negsentIdxs[i] = check_neg_idxs[np.argmax(simis[0])]		
				possent2negimgIdxs[i] = check_neg_idxs[np.argmax(simis[1])]

			new_batch = {"data":[]}
			imgids = {}
			for i in xrange(len(batch_data['pos'])):
				thisPos = batch_data['pos'][i]
				negImg = batch_data['neg'][possent2negimgIdxs[i]][0]
				negSent = batch_data['neg'][posimg2negsentIdxs[i]]
				thisNeg = (negImg,negSent[1],negSent[2])
				new_batch['data'].append((thisPos,thisNeg))
				# get all imageid
				imgids[thisPos[0]] = 1
				imgids[thisNeg[0]] = 1
			# no need to get feature again
			#imgid2idx = {}
			#for imgid in imgids:
			#	imgid2idx[imgid] = len(imgid2idx.keys())
			#image_feats = load_feats(imgid2idx,train_data.shared,config)
			new_batch['imgid2idx'] = imgid2idx
			new_batch['imgidx2feat'] = image_feats
			batch = batchIdx,Dataset(new_batch,"train",shared=train_data.shared,is_train=True,imgfeat_dim=config.feat_dim)

			loss,train_op = trainer.step(sess,batch)
			#print mcb1
			#print "-"*40

		if global_step % save_period != 0: # time to save model
			saver.save(sess,config.save_dir_model,global_step=global_step)
		str_ = "best eval on val %s: %s at %s step, final step %s %s is %s"%(metric,best[metric],best['step'], global_step,metric,finalperf[metric])
		print str_
		self_summary_strs.append(str_)
		if config.write_self_sum:
			f = open(config.self_summary_path,"w")
			f.writelines("%s"%("\n".join(self_summary_strs)))
			f.close()
		if config.record_val_perf:
			pickle.dump(val_perf,open(config.val_perf_path,"wb"))



def test(config):
	if config.is_test_on_val:
		test_data = read_data(config,'val',True,subset=False)
		print "total val samples:%s"%test_data.num_examples
	else:
		test_data = read_data(config,'test',True,subset=False) # here will load shared.p from config.outpath (outbase/modelname/runId/)
		print "total test samples:%s"%test_data.num_examples
	config.imgfeat_dim = test_data.imgfeat_dim
	# get the max_sent_size and other stuff
	print "threshold setting--"
	config_vars = vars(config)
	print "\t"+ " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.thresmeta])

	# cap the numbers
	update_config(config,[test_data],showMeta=True)


	print "renewed ----"
	print "\t" + " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.maxmeta])



	model = get_model(config)

	

	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True # this way it will only allocate nessasary gpu, not take all
	# or you can set hard limit
	#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4



	with tf.Session(config=tfconfig) as sess:
		initialize(load=True,load_best=config.load_best,model=model,config=config,sess=sess)

		if config.is_pack_model:
			saver = tf.train.Saver()
			global_step = model.global_step
			# put input and output to a universal name for reference when in deployment
				# find the nessary stuff in model.get_feed_dict

			# multiple input
			tf.add_to_collection("sents",model.sents)
			tf.add_to_collection("sents_c",model.sents_c)
			tf.add_to_collection("sents_mask",model.sents_mask)

		
			tf.add_to_collection("pis",model.pis)

			# image and text feature
			tf.add_to_collection("image_emb_mat",model.image_emb_mat)
			tf.add_to_collection("existing_emb_mat",model.existing_emb_mat)

			# for getting the highest ranked photo
			#tf.add_to_collection("att_logits",model.att_logits)
			
			tf.add_to_collection("is_train",model.is_train) # TODO, change this to a constant 
			tf.add_to_collection("output",model.s)
			# also save all the model config and note into the model
			pack_model_note = tf.get_variable("model_note",shape=[],dtype=tf.string,initializer=tf.constant_initializer(config.pack_model_note),trainable=False)
			full_config = tf.get_variable("model_config",shape=[],dtype=tf.string,initializer=tf.constant_initializer(json.dumps(vars(config))),trainable=False)

			print "saving packed model"
			# the following wont save the var model_note, model_config that's not in the graph, 
			# TODO: fix this
			"""
			# put into one big file to save
			input_graph_def = tf.get_default_graph().as_graph_def()
			#print [n.name for n in input_graph_def.node]
			 # We use a built-in TF helper to export variables to constants
			output_graph_def = tf.graph_util.convert_variables_to_constants(
				sess, # The session is used to retrieve the weights
				input_graph_def, # The graph_def is used to retrieve the nodes 
				[tf.get_collection("output")[0].name.split(":")[0]] # The output node names are used to select the usefull nodes
			) 
			output_graph = os.path.join(config.pack_model_path,"final.pb")
			# Finally we serialize and dump the output graph to the filesystem
			with tf.gfile.GFile(output_graph, "wb") as f:
				f.write(output_graph_def.SerializeToString())
			print("%d ops in the final graph." % len(output_graph_def.node))
			"""
			# save it into a path with multiple files
			saver.save(sess,
				os.path.join(config.pack_model_path,"final"),
				global_step=global_step)
			print "model saved in %s"%(config.pack_model_path)
			return 

		if config.is_save_weights:
			weight_dict = {}
			weight_sum = open(os.path.join(config.weights_path,"all.txt"),"w")
			for var in tf.trainable_variables():
				shape = var.get_shape()
				weight_sum.writelines("%s %s\n"%(var.name,shape))
				var_val = sess.run(var)
				weight_dict[var.name] = var_val

			np.savez(os.path.join(config.weights_path,"weights.npz"),**weight_dict)
			weight_sum.close()

		last_time = time.time()
		# num_epoch should be 1
		num_steps = int(math.ceil(test_data.num_examples/float(config.batch_size)))*config.num_epochs

		# load the graph and variables
		tester = Tester(model,config,sess)

		perf = evaluate(test_data,config,sess,tester)

	print "performance:"
	print perf
	metric = ["r@1","r@5","r@10","mr"]
	metrics = []
	for one in metric:
		metrics.append("%s_%s"%("t2i",one))
		metrics.append("%s_%s"%("i2t",one))
	print "\t".join([m for m in metrics])
	print "\t".join(["%.4f"%perf[m] for m in metrics])
	if config.save_answers:
		pickle.dump(imgid2sents,open("%s/answers_i2t.p"%config.val_path,"w"))
		pickle.dump(sentId2imgs,open("%s/answers_t2i.p"%config.val_path,"w"))
	


def initialize(load,load_best,model,config,sess):
	tf.global_variables_initializer().run()
	if load:
		#print len(tf.global_variables())
		#print [var.name for var in tf.global_variables()]
		# var_name to the var object
		vars_ = tf.global_variables()
		adams = ["Adam","beta1_power","beta2_power","Adam_1","Adadelta_1","Adadelta"]
		vars_ = [var for var in vars_ if var.name.split(":")[0].split("/")[-1] not in adams]

		ignore = "global_step"
		vars_ = {var.name.split(":")[0]: var for var in vars_ if ignore not in var.name}


		#variable_averages = tf.train.ExponentialMovingAverage(self.config.var_ema_decay)
		#vars_ = variable_averages.variables_to_restore()
		"""
		ema = model.var_ema
		#print len(tf.trainable_variables())
		#print [var.name for var in tf.trainable_variables()]
		if config.load_ema:
			for var in tf.trainable_variables():
				del vars_[var.name.split(":")[0]] # delete the original var name:var
				#print ema.average_name(var)
				vars_[ema.average_name(var)] = var # use the var's moving average version (ema_name:var)
		"""
		saver = tf.train.Saver(vars_, max_to_keep=5)

		load_from = None
		if load_best:
			load_from = config.save_dir_best
		else:
			load_from = config.save_dir
		if config.load_from is not None:
			load_from = config.load_from

		# load the lateste model
		ckpt = tf.train.get_checkpoint_state(load_from)
		if ckpt and ckpt.model_checkpoint_path:
			loadpath = ckpt.model_checkpoint_path
			saver.restore(sess, loadpath)
			print "Model:"
			print "\tloaded %s"%loadpath
			print ""
		else:
			if os.path.exists(load_from): # load_from should be a single .ckpt file
				saver.restore(sess,load_from)
			else:
				raise Exception("Model not exists")



# https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
def cal_total_param():
	total = 0
	for var in tf.trainable_variables():
		shape = var.get_shape()
		var_num = 1
		for dim in shape:
			var_num*=dim.value
		total+=var_num
	return total


if __name__ == "__main__":
	config = get_args()
	# some useful info of the dataset
	config.thresmeta = (
		"sent_size_thres",
		"word_size_thres"
	)
	config.maxmeta = (
		"max_sent_size",
		"max_word_size"
	)
	if config.is_train:
		train(config)
	else:
		test(config)
