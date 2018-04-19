# coding=utf-8
# tensorflow model graph 


import tensorflow as tf
from utils import flatten,reconstruct,Dataset,exp_mask
import numpy as np
import random,sys

VERY_NEGATIVE_NUMBER = -1e30

def get_model(config):
	# implement a multi gpu model?
	with tf.name_scope(config.modelname), tf.device("/gpu:0"):
		# config.modelname = "dan"
		# so,
		# "model_%s"%config.modelname = "model_dan"
		model = Model(config,"model_%s"%config.modelname)

	return model


from copy import deepcopy # for C[i].insert(Y[i])

# all NN layer need a scope variable to avoid name conflict since we may call the function multiple times

# a flatten and reconstruct version of softmax
def softmax(logits,scope=None):
	with tf.name_scope(scope or "softmax"): # noted here is name_scope not variable
		flat_logits = flatten(logits,1)
		flat_out = tf.nn.softmax(flat_logits)
		out = reconstruct(flat_out,logits,1)
		return out


# softmax selection?
# return target * softmax(logits)
# target: [ ..., J, d]
# logits: [ ..., J]
# so [N,M,dim] * [N,M] -> [N,dim], so [N,M] is the attention for each M
# return: [ ..., d] # so the target vector is attended with logits' softmax
# [N,M,JX,JQ,2d] * [N,M,JX,JQ] (each context to query's mapping) -> [N,M,JX,2d] # attened the JQ dimension
def softsel( target, logits, hard=False, hardK=None, scope=None):	
	# logits = [N, J]   = [batch, voc_size]
	# target = [N,J,D] = [batch, voc_size, 512]	
	with tf.variable_scope(scope or "softsel"): # there is no variable to be learn here
		
		# hard attention, will only leave topk weights
		if hard: # False, 패스
			assert hardK > 0
			logits = leaveK(logits,hardK,scope="%s_topk"%(scope or "softsel"))
		
		a = softmax(logits) # shape is the same
		# a : <bound method Tensor.get_shape of <tf.Tensor 'dan/dual_attention/v_att_1/softsel/softmax/Reshape_1:0' shape=(?, ?) dtype=float32>>
		# 아마도 [N, J]가 나와야할듯 왜냐면 text attention이니 어떤 단어가 attention 되어 있는지의 결과를 알려면 
		#     전체 voc size크기의 안에서 weight값이 존재해야한다고 생각
		
		target_rank = len(target.get_shape().as_list())
		# ? target_rank = 3 : 3차원 tensor(=shape 3 차원)의 길이는 3
		
		# [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
		# a x target 하려고 보니,
		#   a = [N, J]
		#   target = [N, J, D]		
		#   그래서, a를 tf.expand_dims(a,-1) : (?, ?, 1) # 끝에 하나 늘려서 계산이 가능하도록 함
		#   이는, attention된 단어들에 대해 weight하겠다는 의미이고
		#  	 a는 attention weight
		#   
		return tf.reduce_sum(tf.expand_dims(a,-1)*target,target_rank-2) # second last dim


# x -> [Num,JX,W,embedding dim] # conv2d requires an input of 4d [batch, in_height, in_width, in_channels]
def conv1d(x,filter_size,height,keep_prob,is_train=None,wd=None,scope=None):
	with tf.variable_scope(scope or "conv1d"):
		num_channels = x.get_shape()[-1] # embedding dim[8]
		filter_var = tf.get_variable("filter",shape=[1,height,num_channels,filter_size],dtype="float")
		bias = tf.get_variable('bias',shape=[filter_size],dtype='float')
		strides = [1,1,1,1]
		# add dropout to input
		
		d = tf.nn.dropout(x,keep_prob=keep_prob)
		outd = tf.cond(is_train,lambda:d,lambda:x)
		#conv
		xc = tf.nn.relu(tf.nn.conv2d(outd,filter_var,strides,padding='VALID')+bias)
		# simple max pooling?
		out = tf.reduce_max(xc,2) # [-1,JX,num_channel]

		if wd is not None:
			add_wd(wd)

		return out

def batch_norm(x,scope=None,is_train=True,epsilon=1e-5,decay=0.9):
	scope = scope or "batch_norm"
	# what about tf.nn.batch_normalization
	return tf.contrib.layers.batch_norm(x,decay=decay,updates_collections=None,epsilon=epsilon,scale=True,is_training=is_train,scope=scope)

def layer_norm(x,epsilon=1e-6,scope="layer_norm"):
	with tf.variable_scope(scope):
		d = x.get_shape()[-1]
		scale = tf.get_variable("layer_norm_scale",[d],initializer=tf.ones_initializer())
		bias = tf.get_variable("layer_norm_bias",[d],initializer=tf.zeros_initializer())

		mean = tf.reduce_mean(x,axis=[-1],keep_dims=True)
		var = tf.reduce_mean(tf.square(x - mean),axis=[-1],keep_dims=True)
		norm_x = (x-mean)*tf.rsqrt(var + epsilon)

		return norm_x*scale + bias


# fully-connected layer
# simple linear layer, without activatation # remember to add it
# [N,M,JX,JQ,2d] => x[N*M*JX*JQ,2d] * W[2d,output_size] -> 
def linear(x,output_size,scope,add_tanh=False,wd=None,bn=False,bias=False,is_train=None,ln=False):
	# bn -> batch norm
	# ln -> layer norm
	with tf.variable_scope(scope):
		# since the input here is not two rank, we flat the input while keeping the last dims
		keep = 1
		#print x.get_shape().as_list()
		flat_x = flatten(x,keep) # keeping the last one dim # [N,M,JX,JQ,2d] => [N*M*JX*JQ,2d]
		#print flat_x.get_shape() # (?, 200) # wd+cwd
		bias_start = 0.0
		if not (type(output_size) == type(1)): # need to be get_shape()[k].value
			output_size = output_size.value

		# add batch_norm
		if bn:
			assert is_train is not None
			flat_x = batch_norm(flat_x,scope="bn",is_train=is_train)

		if ln:
			flat_x = layer_norm(flat_x,scope="ln")


		#print [flat_x.get_shape()[-1],output_size]

		W = tf.get_variable("W",dtype="float",initializer=tf.truncated_normal([flat_x.get_shape()[-1].value,output_size],stddev=0.1))
		flat_out = tf.matmul(flat_x,W)

		if bias:
			bias = tf.get_variable("b",dtype="float",initializer=tf.constant(bias_start,shape=[output_size]))
			flat_out += bias

		if add_tanh:
			flat_out = tf.tanh(flat_out,name="tanh")

		#flat_out = tf.nn.dropout(flat_out,keep_prob)

		if wd is not None:
			add_wd(wd)

		out = reconstruct(flat_out,x,keep)
		return out




# add current scope's variable's l2 loss to loss collection
def add_wd(wd,scope=None):
	if wd != 0.0:
		scope = scope or tf.get_variable_scope().name
		vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		with tf.variable_scope("weight_decay"):
			for var in vars_:
				weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="%s/wd"%(var.op.name))
				tf.add_to_collection("losses",weight_decay)


# needed for tf initialization from numpy array, need a callable
def get_initializer(matrix):
	def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
	return _initializer

# u_t -> hs [N,J,d], m_u[N,d]
# u_t는 text feature로써 bi-lstm의 output (batch,문장안의 word size = vocabulary sze, 512)
# m_u는 메모리벡터 = (batch, 512)
# 기본적으로 위의 두 입력값을 받는 각각의 fc로 구성되어 있음.
# u_t_mask = self.sents_mask 
#   - sents_mask = 초기에 np.zeros([N,J],dtype="bool") = (batch, 문장안의 word size = vocabulary sze)
#   - vocabulary 사이즈가 J이면 학습셋 문장에 존재 할때, 1 아니면 0
#   - 각 워드는 J크기안의 idx값이 존재
#   - 예를 들어, 
#   - 전체 voc_size = 10 이고, 데이터(학습셋) 문장이 "hey boy" 라면, hey의 고유 넘버 = 0, boy의 고유 넘버 = 8이고
#   - u_t_mask는 [1, 0, 0, 0, 0, 0, 0, 0, 1, 0] 이런 형태
def T_att(u_t, m_u, u_t_mask, wd=None, scope=None ,reuse=False, use_concat=False, bn=False, is_train=None, keep_prob=None):
	# u_t : (?, ?, 512)
	# m_u : (?, 512)
	with tf.variable_scope(scope or "t_att"):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		J = tf.shape(u_t)[1] # 문장안의 word size = vocabulary sze, Tensor("dan/dual_attention/t_att_1/strided_slice:0", shape=(), dtype=int32, device=/device:GPU:0)
		d = m_u.get_shape()[-1] # 512		
			
		# use concat to get attention logit
		if use_concat: # False, 사용안함
			# tile m_u first
			m_u_aug = tf.tile(tf.expand_dims(m_u,1),[1,J,1])
			a_u = linear(tf.concat([u_t*m_u_aug,(u_t-m_u_aug)*(u_t-m_u_aug)],2),add_tanh=True,output_size=1,scope="att_logits",bn=bn,ln=False,is_train=is_train)
		else:
			if keep_prob is not None:
				u_t = tf.nn.dropout(u_t,keep_prob)
			# 기본적으로 위의 두 입력값(u_t, m_u)을 받는 각각의 fc로 구성되어 있음.
			W_u = linear(u_t,add_tanh=True, ln=False, output_size=d,wd=wd,scope="W_u",bn=bn,is_train=is_train) # [N,J,d]
			# W_u : <bound method Tensor.get_shape of <tf.Tensor 'dan/dual_attention/t_att_0_1/W_u/Reshape_1:0' shape=(?, ?, 512) dtype=float32>>
			# W_u = [N,J,d] = (batch, 문장안의 word size = vocabulary sze, 512)
			
			if keep_prob is not None:
				m_u = tf.nn.dropout(m_u,keep_prob)
			W_u_m =linear(m_u,add_tanh=True,ln=False,output_size=d,wd=wd,scope="W_u_m",bn=bn,is_train=is_train)# [N,d]
			# W_u_m : <bound method Tensor.get_shape of <tf.Tensor 'dan/dual_attention/t_att_0_1/W_u_m/Reshape_1:0' shape=(?, 512) dtype=float32>>
			# W_u_m = [N,d] = (batch, 512)
			
			W_u_m = tf.tile(tf.expand_dims(W_u_m,1),[1,J,1])
			# (batch, 512) -> (batch, 문장안의 word size = vocabulary sze, 512)
			# <bound method Tensor.get_shape of <tf.Tensor 'dan/dual_attention/t_att_1_1/Tile:0' shape=(?, ?, 512) dtype=float32>>
			
			h_u = W_u * W_u_m #[N,J,d]
			# (?, ?, 512) > (batch, 문장안의 word size = vocabulary sze, 512)
			
			# [N,J,1] -> softmax 넣기 위한
			a_u = linear(h_u,output_size=1,ln=False,wd=wd,scope="W_u_h",add_tanh=False,bias=False,bn=bn,is_train=is_train)
		# [N,J,1] -> [N,J]
		a_u = tf.squeeze(a_u,2)
			
		# ? 단어가 존재하는 index와 일치하는 곳의 value만 남겨두겠다는 심산인듯~
		#    u_t_mask = [1,   0,   0, 0,  0, 0,  0, 0, 1, 0]
		#    a_u      = [1.2, 3, 4.3, 6,  8, 9, 19, 1, 2, 0]
		#        so,    [1.2, 0, 0, 0,  0, 0, 0, 0, 2, 0] = exp_mask( a_u, u_t_mask)
		#              > 실제 0은 -1e30 으로 채워지는듯 
		# u_t_mask : <bound method Tensor.get_shape of <tf.Tensor 'dan/sents_neg_mask:0' shape=(?, ?) dtype=bool>>
		a_u = exp_mask(a_u, u_t_mask)

		# a_u = [N,J]
		# u_t = [N,J,D] = (batch,문장안의 word size = vocabulary sze, 512)
		# [N,d]		
		u = softsel(u_t, a_u, hard=False)
	return u

#v_t -> [N,L,idim]
# m_v -> [N,d]
# V_att와 거의 동일
def V_att(v_t,m_v,wd=None,scope=None,reuse=False,use_concat=False,bn=False,is_train=None,keep_prob=None):
	
	# v_t : image vector 
	#       (?, ?, 2048) > ? (batch, 14x14, 2048)
	# m_v : image 관련 memory vector
	#       (?, 512) > (batch, 512)
	
	with tf.variable_scope(scope or "v_att"):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		d = m_v.get_shape()[-1] # 512
		L = tf.shape(v_t)[1] # 14x14? , Tensor("dan/dual_attention/v_att_1_1/strided_slice:0", shape=(), dtype=int32, device=/device:GPU:0)

		if use_concat: # False, 그냥 패스
			# tile m_v first
			m_v_aug = tf.tile(tf.expand_dims(m_v,1),[1,L,1])
			v_t_tran = linear(v_t,ln=False,add_tanh=True,output_size=d,wd=wd,scope="W_v",bn=bn,is_train=is_train) # [N,L,d]
			a_v = linear(tf.concat([v_t_tran*m_v_aug,(v_t_tran-m_v_aug)*(v_t_tran-m_v_aug)],2),ln=False,add_tanh=True,output_size=1,scope="att_logits",bn=bn,is_train=is_train)
			a_v = tf.squeeze(a_v,2)		
		else:
			if keep_prob is not None:
				v_t = tf.nn.dropout(v_t,keep_prob)				
			
			W_v = linear(v_t, ln=False, add_tanh=True, output_size=d, wd=wd, scope="W_v", bn=bn, is_train=is_train) # [N,L,d]
			# v_t = (batch, 14x14, 2048)
			# W_v = (?, ?, 512) >  (batch, 14x14, 512)
			
			if keep_prob is not None:
				m_v = tf.nn.dropout(m_v,keep_prob)
				
			W_v_m =linear(m_v,ln=False,add_tanh=True,output_size=d,wd=wd,scope="W_v_m",bn=bn,is_train=is_train)# [N,d]
			# m_v   = (batch, 512)
			# W_v_m =  (?, 512) > (batch, 512)
			
			W_v_m = tf.tile(tf.expand_dims(W_v_m,1),[1,L,1])
			# W_v_m = <bound method Tensor.get_shape of <tf.Tensor 'dan/dual_attention/v_att_1_1/Tile:0' shape=(?, ?, 512) dtype=float32>>
			# 마찬가지로 element wise mul ,  두 tensor 모두 같은 차원으로 만들어야한다.
			# 	(batch, 14x14, 512) > 512 차원은 같은 값을 가진 벡터를 14x14 개 만든다.
			
			h_v = W_v * W_v_m #[N,L,d],  
			#  element wise mul 			
			#  h_v = <bound method Tensor.get_shape of <tf.Tensor 'dan/dual_attention/v_att_1_1/mul:0' shape=(?, ?, 512) dtype=float32>>
			
		
			# [N,L,1]
			a_v = linear(h_v, ln=False,output_size=1,wd=wd,add_tanh=False,scope="W_v_h",bn=bn,bias=False,is_train=is_train)
			# a_v = (?, ?, 1)
			
			a_v = tf.squeeze(a_v,2)
			# a_v = (?, ?)
		# V_att와 거의 동일
		v = softsel(v_t,a_v,hard=False) #[N,L,idim]
		v = linear(v,ln=False,add_tanh=True,output_size=d,wd=wd,scope="P_v",bn=bn,is_train=is_train)
		
	return v 

class Model():
	def __init__(self,config,scope):
		
		# model_dan
		self.scope  = scope
		
		# config의 내용
		# Namespace(batch_norm=False, batch_size=256, char_count_thres=10, char_emb_size=8, char_out_size=100, 
		# char_vocab_size=47, clip_gradient_norm=0.1, concat_att=False, concat_rnn=False, feat_dim=[14, 14, 2048], 
		# featpath='resnet-152/', finetune_wordvec=False, hidden_size=512, hn_num=32, ignore_vars=None, imgfeat_dim=[14, 14, 2048], 
		# init_lr=0.1, is_pack_model=False, is_save_vis=False, is_save_weights=False, is_test=False, is_test_on_val=False, 
		# is_train=True, keep_prob=0.5, learning_rate_decay=0.95, learning_rate_decay_examples=500000, load=False, load_best=False, 
		# load_from=None, margin=100.0, max_sent_size=82, max_word_size=16, maxmeta=('max_sent_size', 'max_word_size'), modelname='dan', 
		# no_wordvec=True, num_epochs=60, num_hops=2, optimizer='momentum', outbasepath='models', outpath='models/dan/00', 
		# pack_model_note=None, pack_model_path=None, prepropath='prepro', record_val_perf=True, runId=0, save_answers=False, 
		# save_dir='models/dan/00/save', save_dir_best='models/dan/00/best', save_dir_best_model='models/dan/00/best/save-best', 
		# save_dir_model='models/dan/00/save/save', save_period=1000, self_summary_path='models/dan/00/train_sum.txt', sent_size_thres=200, 
		# thresmeta=('sent_size_thres', 'word_size_thres'), use_char=False, val_path='', val_perf_path='models/dan/00/val_perf.p', wd=0.0005, 
		# word_count_thres=1, word_emb_size=512, word_size_thres=20, word_vocab_size=11798, write_self_sum=True)
		self.config = config
		
		
		# a step var to keep track of current training process
		self.global_step = tf.get_variable('global_step',shape=[],dtype='int32',initializer=tf.constant_initializer(0),trainable=False) # a counter

		# get all the dimension here
		#N = self.N = config.batch_size
		N = self.N = None # batch size
		
		VW = self.VW = config.word_vocab_size # 11798
		VC = self.VC = config.char_vocab_size # 47
		W  = self.W = config.max_word_size    # 16

		# embedding dim
		self.cd, self.wd, self.cwd = config.char_emb_size,config.word_emb_size,config.char_out_size
		# self.cd = 8 , char_emb_size
		# self.wd = 512 , word_emb_size
		# self.cwd = 100 , char_out_size(?)

		# image dimension
		self.idim = config.imgfeat_dim # [14, 14, 2048] : conv feature map
	
		
		# Tensor("dan/Const:0", shape=(), dtype=int32, device=/device:GPU:0)
		self.img_att_logits = tf.constant(-1) # the 3d attention logits
		# Tensor("dan/Const_1:0", shape=(), dtype=int32, device=/device:GPU:0)
		self.sent_att_logits = tf.constant(-1) # the question attention logits if there is 
		
		# N = batch size
		self.sents      = tf.placeholder('int32',[N, None],name="sents") # 문장
		self.sents_c    = tf.placeholder("int32",[N, None, W],name="sents_c") # charactor, W = self.W = config.max_word_size = 16
		self.sents_mask = tf.placeholder("bool",[N, None],name="sents_mask") # to get the sequence length

		self.pis        = tf.placeholder('int32',[N],name="pis") # ?


		# for training - 이게 어떤 의미?
		self.pis_neg        = tf.placeholder('int32',[N],name="pis_neg") # ?
		self.sents_neg      = tf.placeholder('int32',[N, None],name="sents_neg") # 문장
		self.sents_neg_c    = tf.placeholder("int32",[N, None, W],name="sents_neg_c") # charactor, W = self.W = config.max_word_size = 16
		self.sents_neg_mask = tf.placeholder("bool",[N, None],name="sents_neg_mask") # to get the sequence length

		
		# feed in the pretrain word vectors for all batch
		# config.word_emb_size = 512
		self.existing_emb_mat = tf.placeholder('float',[None, config.word_emb_size], name="pre_emb_mat")

		# feed in the image feature for this batch
		# [photoNumForThisBatch,image_dim]
		# now image feature could be a conv feature tensor instead of vector
		#self.image_emb_mat = tf.placeholder("float",[None,config.imgfeat_size],name="image_emb_mat")
		self.image_emb_mat = tf.placeholder("float",[None]+config.imgfeat_dim,name="image_emb_mat")
		# [None]+config.imgfeat_dim = [None, 14, 14, 2048]

		# used for drop out switch
		self.is_train = tf.placeholder('bool', [], name='is_train')

		# forward output
		# the following will be added in build_forward and build_loss()
		self.logits = None

		self.yp = None # prob

		self.loss = None

		self.build_forward()
		self.build_loss()

	def build_forward(self):
		
		config = self.config
		VW = self.VW
		VC = self.VC
		W  = self.W
		N  = self.N
		
		# VW = 11798, word_vocab_size 
		# VC = 47, char_vocab_size 
		# W  = 16, max_word_size    
		# N  = None, batch size
		
		J = tf.shape(self.sents)[1] # sentence size, Tensor("dan/strided_slice:0", shape=(), dtype=int32, device=/device:GPU:0) 
		d = config.hidden_size # 512
		
		if config.concat_rnn: # False
			d = 2*d
		# d : 512
		
		# embeding size
		cdim, wdim, cwdim = self.cd, self.wd, self.cwd 
		# cdim  = 8 , config.char_emb_size
		# wdim  = 512 , config.word_emb_size
		# cwdim = 100, cwd: config.char_out_size

		# image feature dim
		idim = self.idim # image_feat dimension # it is a list, [1536] or [8,8,1536]
		# 실제로는 [14, 14, 2048]

		# embedding
		with tf.variable_scope('emb'):
			
			# char stuff
			if config.use_char: # false, 그래서 여긴 패스
			#with tf.variable_scope("char"):
				# [char_vocab_size,char_emb_dim]
				with tf.variable_scope("var"), tf.device("/cpu:0"): 
					char_emb = tf.get_variable("char_emb",shape=[VC,cdim],dtype="float")

				# the embedding for each of character 
				# [N,J,W,cdim]
				Asents_c     = tf.nn.embedding_lookup(char_emb,self.sents_c)
				Asents_neg_c = tf.nn.embedding_lookup(char_emb,self.sents_neg_c)
				
				#char CNN
				filter_size = cwdim # output size for each word
				filter_height = 5
				#[N,J,cwdim]
				with tf.variable_scope("conv"):
					xsents = conv1d(Asents_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					tf.get_variable_scope().reuse_variables()
					xsents_neg = conv1d(Asents_neg_c,filter_size,filter_height,config.keep_prob,self.is_train,wd=config.wd,scope="conv1d")
					
					
			# word stuff
			with tf.variable_scope('word'):
				
				with tf.variable_scope("var"), tf.device("/cpu:0"):
					# get the word embedding for new words
					if config.is_train: # True
						# for new word
						if config.no_wordvec: # True
							# [VW, wdim] = [11798, 512] = [word_vocab_size, word_emb_size]
							word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW, wdim],initializer=tf.truncated_normal_initializer(stddev=1.0))
						else:
							word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW,wdim],initializer=get_initializer(config.emb_mat)) # it's just random initialized, but will include glove if finetuning
					else: # save time for loading the emb during test
						word_emb_mat = tf.get_variable("word_emb_mat",dtype="float",shape=[VW,wdim])
					# concat with pretrain vector
					# so 0 - VW-1 index for new words, the rest for pretrain vector
					# and the pretrain vector is fixed
					# config.finetune_wordvec = False & config.no_wordvec = True, 그래서 여긴 패스
					if not config.finetune_wordvec and not config.no_wordvec:
						word_emb_mat = tf.concat([word_emb_mat,self.existing_emb_mat],0)

				#[N,J,wdim]
				# word_emb_mat   = <tf.Variable 'emb/word/var/word_emb_mat:0' shape=(11798, 512) dtype=float32_ref>
				# self.sents     = Tensor("dan/sents:0", shape=(?, ?), dtype=int32, device=/device:GPU:0)
				# self.sents_neg = Tensor("dan/sents_neg:0", shape=(?, ?), dtype=int32, device=/device:GPU:0)
				Asents = tf.nn.embedding_lookup(word_emb_mat,self.sents)				 
				Asents_neg = tf.nn.embedding_lookup(word_emb_mat,self.sents_neg)
				
				"""
				# need one-hot representation of sents
				Asents = linear(self.sents,output_size=wdim,scope="word_emb",bias=False,add_tanh=False)
				tf.get_variable_scope().reuse_variables()
				Asents_neg = linear(self.sents_neg,output_size=wdim,scope="word_emb",bias=False,add_tanh=False)
				"""
			
			# concat char and word
			if config.use_char: # False , 그래서 여긴 패스
				xsents = tf.concat([xsents,Asents],2)
				xsents_neg = tf.concat([xsents_neg,Asents_neg],2)
			else:
				xsents = Asents
				xsents_neg = Asents_neg

			# get the image feature
			with tf.variable_scope("image"):

				# [N] -> [N,idim]
				# [N] -> [N,8,8,1536] if using conv feature
				
				# Tensor("dan/emb/image/strided_slice:0", shape=(), dtype=int32, device=/device:GPU:0)
				NP = tf.shape(self.pis)[0] 
				
				# self.image_emb_mat : Tensor("dan/image_emb_mat:0", shape=(?, 14, 14, 2048), dtype=float32, device=/device:GPU:0)
				# self.pis : Tensor("dan/pis:0", shape=(?,), dtype=int32, device=/device:GPU:0)
				xpis = tf.nn.embedding_lookup(self.image_emb_mat,self.pis)
				#tf.get_variable_scope().reuse_variables()
				
				# self.pis_neg: Tensor("dan/pis_neg:0", shape=(?,), dtype=int32, device=/device:GPU:0)
				xpis_neg = tf.nn.embedding_lookup(self.image_emb_mat,self.pis_neg)
					
				# self.idim = [14, 14, 2048]
				# 그래서 [-1]은 마지막 여기선 [2] = 2048
				xpis     = tf.reshape(xpis,[NP,-1,self.idim[-1]]) # (?, ?, 2048)
				xpis_neg = tf.reshape(xpis_neg,[NP,-1,self.idim[-1]])
					
			
		# not used by the paper
		"""
		with tf.variable_scope("input_layer_norm"):
			xsents = layer_norm(xsents,scope="xsents_ln")
			xpis = layer_norm(xpis,scope="xpis_ln")

			tf.get_variable_scope().reuse_variables()

			xsents_neg = layer_norm(xsents_neg,scope="xsents_ln")
			xpis_neg = layer_norm(xpis_neg,scope="xpis_ln")
		"""
		
		############################################################################################################
		# Text Representation
		#  - 기본적으로 Bi-LSTM 적용, 2 f/b 의 결과를 concat함.
		############################################################################################################
		# LSTM / GRU?
		# config.hidden_size = 512
		cell_text = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,state_is_tuple=True)
		#cell_text = tf.nn.rnn_cell.GRUCell(d)
		# add dropout
		# self.is_train : Tensor("dan/is_train:0", shape=(), dtype=bool, device=/device:GPU:0)
		# config.keep_prob = 0.5
		keep_prob = tf.cond(self.is_train,lambda:tf.constant(config.keep_prob),lambda:tf.constant(1.0))
		# keep_prob : Tensor("dan/cond/Merge:0", shape=(), dtype=float32, device=/device:GPU:0)
	
		cell_text = tf.nn.rnn_cell.DropoutWrapper(cell_text,keep_prob)

		# sequence length for each
		sents_len     = tf.reduce_sum(tf.cast(self.sents_mask,"int32"),1) # [N] , N = batch size
		sents_neg_len = tf.reduce_sum(tf.cast(self.sents_neg_mask,"int32"),1) # [N] 
		# self.sents_mask     = Tensor("dan/sents_mask:0", shape=(?, ?), dtype=bool, device=/device:GPU:0)
		# self.sents_neg_mask = Tensor("dan/sents_neg_mask:0", shape=(?, ?), dtype=bool, device=/device:GPU:0)
		# sents_len           = Tensor("dan/Sum:0", shape=(?,), dtype=int32, device=/device:GPU:0)
		# sents_neg_len       = Tensor("dan/Sum_1:0", shape=(?,), dtype=int32, device=/device:GPU:0)

		with tf.variable_scope("reader"):
			with tf.variable_scope("text"):
				(fw_hs, bw_hs),(fw_ls, bw_ls) = tf.nn.bidirectional_dynamic_rnn(cell_text, cell_text, xsents, sequence_length=sents_len, dtype="float", scope="utext")
				# fw_hs = Tensor("dan/reader/text/utext/fw/fw/transpose:0", shape=(?, ?, 512), dtype=float32, device=/device:GPU:0)
				# bw_hs = Tensor("dan/reader/text/ReverseSequence:0", shape=(?, ?, 512), dtype=float32, device=/device:GPU:0)
				# fw_ls = LSTMStateTuple(c=<tf.Tensor 'dan/reader/text/utext/fw/fw/while/Exit_2:0' shape=(?, 512) dtype=float32>, h=<tf.Tensor 'dan/reader/text/utext/fw/fw/while/Exit_3:0' shape=(?, 512) dtype=float32>)
				# bw_ls = LSTMStateTuple(c=<tf.Tensor 'dan/reader/text/utext/bw/bw/while/Exit_2:0' shape=(?, 512) dtype=float32>, h=<tf.Tensor 'dan/reader/text/utext/bw/bw/while/Exit_3:0' shape=(?, 512) dtype=float32>)

				
				# concat the fw and backward lstm output
				#hq = tf.concat([fw_hq,bw_hq],2)
				if config.concat_rnn: # False, 그래서 여긴 패스
					hs = tf.concat([fw_hs,bw_hs],2)
					ls = tf.concat([fw_ls.h,bw_ls.h],2)  # 사용하지않음
				else:
					# this is the paper
					hs = fw_hs+bw_hs 
					ls = fw_ls.h+bw_ls.h # 사용하지않음
				# hs : Tensor("dan/reader/text/add:0", shape=(?, ?, 512), dtype=float32, device=/device:GPU:0)
				# addition, same as the paper

				#lq = tf.concat([fw_lq.h,bw_lq.h],1) #LSTM CELL
				#lq = tf.concat([fw_lq,bw_lq],1) # GRU

				tf.get_variable_scope().reuse_variables()

				(fw_hs_neg, bw_hs_neg),(fw_ls_neg, bw_ls_neg) = tf.nn.bidirectional_dynamic_rnn(cell_text,cell_text,xsents_neg,sequence_length=sents_neg_len,dtype="float",scope="utext")
				if config.concat_rnn:
					hs_neg = tf.concat([fw_hs_neg,bw_hs_neg],2)
					ls_neg = tf.concat([fw_ls_neg.h,bw_ls_neg.h],2) # 사용하지않음
				else:
					hs_neg = fw_hs_neg+bw_hs_neg
					ls_neg = fw_ls_neg.h+bw_ls_neg.h # 사용하지않음

			# config.wd = 0.0005
			if config.wd is not None: # l2 weight decay for the reader
				add_wd(config.wd)

		if config.concat_rnn: # False, , 그래서 여긴 패스
			d = 2*d
		# d = 512
		# N = batch size
		# J = # sentence size, Tensor("dan/strided_slice:0", shape=(), dtype=int32, device=/device:GPU:0)
		# d = 512
		##
		# hs [N,J,d] = (?, ?, 512)
		# hs_neg [N,J,d]
		##
		# 마지막 결과인 hs vs xpis는            
		# 여기서 중요한것은 이미지와 텍스트의 최종 dimension(512)을 맞춰주는것이다.
		#   - memrory vector 구하기 위해 [batch, 512]로 평균을 구하면서 맞춰준다.     
		
		# idim = [14, 14, 2048] 
		##
		# xpis [N,L,idim] # (?, ?, 2048)
		# xpis_neg [N,L,idim] # (?, ?, 2048)
		with tf.variable_scope("dual_attention"):
			# for training
			s       = []
			s_v_neg = []
			s_u_neg = []

			# for inferencing
			z_v = []
			z_u = []

			# memory vectors # [N,d]
			# initialization
			############################################################################################################
			# memory vector - init
			#  - visual attention 결과와 text attention 결과를 concat하는 구조
			#  - 초기값은 평균벡터 conv feature와 text bi-lstm의 평균 
			#  - 평균이란 의미는 nx512(batch 제외)를 512 차원으로 줄여야하는데 그럴려면 n을 1 dim으로, 즉 평균으로 하여 구하여, 
			#    이를 하나씩, 512번하는 형태인 512(1x512)형태로 만든다. 
			#    그래서, (b,n,512)-> (b,512) 형태로 만들기위해, 
			#              텍스트 정보는 tf.reduce_mean(hs,1): 1 > axis=1이고 n->1               
			#               이미지 정보는 fc(linear)로 (batch, ?, 2048) > (b,512)           
			############################################################################################################
			with tf.variable_scope("mem_init"):
				# text
				# assuming the non-word location is zeros
				# [N,d] / [N]
				
				#u_0 = tf.truediv(tf.reduce_sum(hs,1), tf.expand_dims(tf.cast(sents_len,tf.float32),1))
				#u_0_neg = tf.truediv(tf.reduce_sum(hs_neg,1),tf.expand_dims(tf.cast(sents_neg_len,tf.float32),1))
				
				u_0     = tf.reduce_mean(hs, 1) # [batch, ?, 512] -> [batch, 512]
				u_0_neg = tf.reduce_mean(hs_neg, 1)

				u_0     = tf.nn.dropout(u_0, keep_prob)
				u_0_neg = tf.nn.dropout(u_0_neg, keep_prob)

				#u_0 = ls
				#u_0_neg = ls_neg

				m_u = u_0
				m_u_neg = u_0_neg

				# img
				# [N,L,idim] -> [N,d]
				with tf.variable_scope("img_0"):
					# linear : fully-connected layer
					# xpis : (?, ?, 2048)
					# 실제 이미지정보는 여기서 맞춰준다. (batch, ?, 2048) > (b,512) 
					v_0    = linear(tf.reduce_mean(xpis,1), output_size=d, add_tanh=True,ln=False,bias=True,bn=False,scope="img_p0")
					# v_0 : (?, 512)
					tf.get_variable_scope().reuse_variables()
					v_0_neg = linear(tf.reduce_mean(xpis_neg,1),output_size=d,ln=False,add_tanh=True,bias=True,bn=False,scope="img_p0")

					v_0     = tf.nn.dropout(v_0,keep_prob) # v_0 : (?, 512)					
					v_0_neg = tf.nn.dropout(v_0_neg,keep_prob)

				m_v     = v_0
				m_v_neg = v_0_neg

				z_v.append(v_0)
				z_u.append(u_0)
				
				############################################################################################################
				# embedding - 서로 다른 공간의 feature를 embedding 하면 similarity를 구할수 있다.
				# 두 메모리 벡터에 대해 similarity를 구한다.(논문 그림 참조 - fig.4)
				# 실제 이 정보는(fig.4의 그림) 현 attention 과정이 아니라 next 과정의 입력값으로 들어간다.
				############################################################################################################
				
				# simi K=0
				# get similarity, inner product
				s_0 = tf.reduce_sum(tf.multiply(v_0,u_0),1) #[N] - N batch size
				s.append(s_0)
				# for training
				s_0_v_neg = tf.reduce_sum(tf.multiply(v_0_neg,u_0),1) #[N]
				s_v_neg.append(s_0_v_neg)

				s_0_u_neg = tf.reduce_sum(tf.multiply(v_0,u_0_neg),1) #[N]
				s_u_neg.append(s_0_u_neg)
				
			############################################################################################################
			# attention 
			############################################################################################################
			for i in xrange(config.num_hops):
				
				############################################################################################################
				# text attention
				############################################################################################################
				# u = [N,d] = (?, 512)
				# hs : text encoder(bi-lstm)의 결과 [N, J, D]
				# m_u : text memory vector [N, J]
				u = T_att(hs, m_u,self.sents_mask,use_concat=config.concat_att,wd=config.wd,scope="t_att_%s"%i,bn=config.batch_norm,is_train=self.is_train,keep_prob=keep_prob)
				z_u.append(u)		
				
				############################################################################################################
				# image attention
				############################################################################################################
				# img
				v = V_att(xpis,m_v,wd=config.wd,use_concat=config.concat_att,scope="v_att_%s"%i,bn=config.batch_norm,is_train=self.is_train,keep_prob=keep_prob)
				z_v.append(v)

				# for training

				u_neg = T_att(hs_neg,m_u_neg,self.sents_neg_mask,use_concat=config.concat_att,wd=config.wd,scope="t_att_%s"%i,reuse=True,bn=config.batch_norm,is_train=self.is_train,keep_prob=keep_prob)
				
				v_neg = V_att(xpis_neg,m_v_neg,use_concat=config.concat_att,wd=config.wd,scope="v_att_%s"%i,reuse=True,bn=config.batch_norm,is_train=self.is_train,keep_prob=keep_prob)				

				# get similarity #  for training
				s_i = tf.reduce_sum(tf.multiply(v,u),1) #[N]
				s.append(s_i)
				
				s_i_v_neg = tf.reduce_sum(tf.multiply(v_neg,u),1) #[N]
				s_v_neg.append(s_i_v_neg)

				s_i_u_neg = tf.reduce_sum(tf.multiply(v,u_neg),1) #[N]
				s_u_neg.append(s_i_u_neg)

				# new text memory
				m_u = m_u + u
				m_v = m_v + v

				m_u_neg = m_u_neg + u_neg
				m_v_neg = m_v_neg + v_neg


			# a list of [N,d] -> stack -> [N,hop+1,d]
			z_u = tf.stack(z_u,axis=1) # [N,hop+1,d]
			z_v = tf.stack(z_v,axis=1) # [N,hop+1,d]

			# for training
			s = tf.stack(s,axis=1) #[N,hop+1]

			s_v_neg = tf.stack(s_v_neg,axis=1)
			s_u_neg = tf.stack(s_u_neg,axis=1)


			s = tf.reduce_sum(s,1) # [N]		
			s_v_neg = tf.reduce_sum(s_v_neg,1)
			s_u_neg = tf.reduce_sum(s_u_neg,1)

			# inferencing
			self.z_u = z_u
			self.z_v = z_v
			# for training
			self.s = s
			self.s_v_neg = s_v_neg
			self.s_u_neg = s_u_neg

		

	def build_loss(self):
		# s -> (v,u)
		# s_v_neg -> (v_neg,u)
		# s_u_neg -> (v,u_neg)
		m = self.config.margin
		losses = tf.maximum(0.0,m-self.s+self.s_v_neg) + tf.maximum(0.0,m-self.s+self.s_u_neg) #[N]

		losses = tf.reduce_mean(losses)

		tf.add_to_collection("losses",losses)

		# there might be l2 weight loss in some layer
		self.loss = tf.add_n(tf.get_collection("losses"),name="total_losses")
		#?
		#tf.summary.scalar(self.loss.op.name, self.loss)

	# givng a batch of data, construct the feed dict
	def get_feed_dict(self,batch,is_train=False):
		# each batch will be (imgId,sent,sent_c), 
		# for training, also the negative data
		# for testing, batch will be sent data and image data , will generate two feed_dict

		assert isinstance(batch,Dataset)
		# get the cap for each kind of step first
		config = self.config
		#N = config.batch_size
		N = len(batch.data['data'])
		NP = N

		if not is_train:
			NP = len(batch.data['imgs'])
		
		J = config.max_sent_size # 86
		
		VW = config.word_vocab_size
		VC = config.char_vocab_size
		d = config.hidden_size
		W = config.max_word_size

		if is_train:
			new_J = 0
			for pos,neg in batch.data['data']:
				new_J = max([new_J,len(pos[1]),len(neg[1])])
			J = min(new_J,J)
		else:
			new_J = 0
			for data in batch.data['data']:
				new_J = max([new_J,len(data[0])])
			J = min(new_J,J)

		feed_dict = {}

		# initial all the placeholder
		# all words initial is 0 , means -NULL- token
		sents = np.zeros([N,J],dtype='int32')
		sents_c = np.zeros([N,J,W],dtype="int32")
		sents_mask = np.zeros([N,J],dtype="bool")

		pis = np.zeros([NP],dtype='int32')			

		# link the feed_dict
		feed_dict[self.sents] = sents
		feed_dict[self.sents_c] = sents_c
		feed_dict[self.sents_mask] = sents_mask

		feed_dict[self.pis] = pis
		
		if is_train:
			sents_neg      = np.zeros([N,J],dtype='int32')
			sents_neg_c    = np.zeros([N,J,W],dtype="int32")
			sents_neg_mask = np.zeros([N,J],dtype="bool")

			pis_neg        = np.zeros([NP],dtype='int32')

			feed_dict[self.sents_neg] = sents_neg
			feed_dict[self.sents_neg_c] = sents_neg_c
			feed_dict[self.sents_neg_mask] = sents_neg_mask
			feed_dict[self.pis_neg] = pis_neg
			

		feed_dict[self.image_emb_mat] = batch.data['imgidx2feat']

		feed_dict[self.is_train] = is_train

		
		# this could be empty, when finetuning or not using pretrain word vectors
		if not config.finetune_wordvec and not config.no_wordvec:
			feed_dict[self.existing_emb_mat] = batch.shared['existing_emb_mat']


		def get_word(word):
			d = batch.shared['word2idx'] # this is for the word not in glove
			if d.has_key(word.lower()):
				return d[word.lower()]
			# the word in glove
			
			d2 = batch.shared['existing_word2idx'] # empty for finetuning and no pretrain
			if d2.has_key(word.lower()):
				return d2[word.lower()] + len(d) # all idx + len(the word to train)
			return 1 # 1 is the -UNK-

		def get_char(char):
			d = batch.shared['char2idx']
			if d.has_key(char):
				return d[char]
			return 1

		data = batch.data['data']
		imgid2idx = batch.data['imgid2idx']

		for i in xrange(len(data)):
			if is_train:
				pos,neg = data[i]
				imgid_pos,sent_pos,sent_c_pos = pos
				imgid_neg,sent_neg,sent_c_neg = neg

				pis[i] = imgid2idx[imgid_pos]
				pis_neg[i] = imgid2idx[imgid_neg]

				for j in xrange(len(sent_pos)):
					if j == config.max_sent_size:
						break
					wordIdx = get_word(sent_pos[j])
					sents[i,j] = wordIdx
					sents_mask[i,j] = True

				for j in xrange(len(sent_c_pos)):
					if j == config.max_sent_size:
						break
					for k in xrange(len(sent_c_pos[j])):
						if k == config.max_word_size:
							break
						charIdx = get_char(sent_c_pos[j][k])
						sents_c[i,j,k] = charIdx

				for j in xrange(len(sent_neg)):
					if j == config.max_sent_size:
						break
					wordIdx = get_word(sent_neg[j])
					sents_neg[i,j] = wordIdx
					sents_neg_mask[i,j] = True

				for j in xrange(len(sent_c_neg)):
					if j == config.max_sent_size:
						break
					for k in xrange(len(sent_c_neg[j])):
						if k == config.max_word_size:
							break
						charIdx = get_char(sent_c_neg[j][k])
						sents_neg_c[i,j,k] = charIdx
			else:
				sent,sent_c = data[i]
				for j in xrange(len(sent)):
					if j == config.max_sent_size:
						break
					wordIdx = get_word(sent[j])
					sents[i,j] = wordIdx
					sents_mask[i,j] = True

				for j in xrange(len(sent_c)):
					if j == config.max_sent_size:
						break
					for k in xrange(len(sent_c[j])):
						if k == config.max_word_size:
							break
						charIdx = get_char(sent_c[j][k])
						sents_c[i,j,k] = charIdx

		# for inferecing, image and text are separate
		if not is_train:
			#print "N:%s, img:%s"%(N,len(batch.data['imgs'])) # not the same
			for i in xrange(len(batch.data['imgs'])):
				imgid = batch.data['imgs'][i]
				pis[i] = imgid2idx[imgid]


		#print feed_dict
		return feed_dict


