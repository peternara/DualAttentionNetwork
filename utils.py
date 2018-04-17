# coding=utf-8
# utils for tensorflow

import tensorflow as tf
from operator import mul
from itertools import izip_longest
import random,itertools,operator
from collections import defaultdict
import math
import numpy as np
from tqdm import tqdm
import sys,os,random

def grouper(l,n):
	# given a list and n(batch_size), devide list into n sized chunks
	# last one will fill None
	args = [iter(l)]*n
	out = izip_longest(*args,fillvalue=None)
	out = list(out)
	return out

def sec2time(secs):
	#return strftime("%H:%M:%S",time.gmtime(secs)) # doesnt support millisecs
	
	m,s = divmod(secs,60)
	#print m,s
	h,m = divmod(m,60)
	if(s >= 10.0):
		return "%02d:%02d:%.3f"%(h,m,s)
	else:
		return "%02d:%02d:0%.3f"%(h,m,s)

class Dataset():
	# data should be 
	"""
	data = {
		"data":[]
		"idx":[]
		

	}
	in mini batch,
	data added all other things it need from the shared
	, shared is the whole shared dict

	"""
	# subset for validation
	def __init__(self,data,datatype,shared=None,is_train=True,imgfeat_dim=None):
		self.data = data # data is a dict  {"data":[],"sentids":[],"imgids":[],"sentid2data"}, 
		self.datatype = datatype
		self.shared = shared
		self.is_train = is_train
		if self.shared['imgid2feat'] is not None:
			self.imgfeat_dim = list(self.shared['imgid2feat'][self.shared['imgid2feat'].keys()[0]].shape) # it could be (1536) or conv feature (8,8,1536)
		else:
			assert imgfeat_dim is not None
			self.imgfeat_dim = imgfeat_dim

		#self.data_list = []
		#if not is_train:
			# not train, then should generate full composition of 1000 imageid to 5000 sentids
			# full 1000 takes 20 minutes with batch_size 512 
			# 100 takes 37 seconds with batch_size 64
			# 300 takes 4 minutes with batch_size 64
			# 200 takes 2 minutes with batch_size 64
			#for i in xrange(len(self.data['imgids'])):
		"""
			if subset:
				subs = 400
				#subs = 2
				subs_t = subs*5
				for i in xrange(subs):
					for j in xrange(subs_t):
						self.data_list.append((i,j))
			else:
				for i in xrange(len(self.data['imgids'])):
					for j in xrange(len(self.data['sentids'])):
						self.data_list.append((i,j))
		"""
		#else:
		#	self.data_list = self.data['data']


		self.valid_idxs = range(len(self.data['data'])) #(imgid,si,csi) or (si,csi)

		self.num_examples = len(self.valid_idxs)


	def get_by_idxs(self,idxs):
		# if it is for training, will also sample negative
		out = {"data":[]}
		if self.is_train:
			"""
			# get the idx remaining for sampling
			neg_pool = [idx for idx in self.valid_idxs if idx not in idxs]
			neg_sampled = random.sample(neg_pool,len(idxs)) 
			#neg_pool = [idx for idx in self.valid_idxs if idx not in idxs and not pos_img_ids.has_key(self.data['data'][idx][0])]
			# this sampling has potential problem? negative are paired?

			# randomly decide to flip the whole batch during training
			flip = False
			#ran = random.random()
			#if ran > 0.9:
			#	flip=True

			out["flip"] = flip

			#out = defaultdict(list) # so the initial value is a list
			for i in xrange(len(idxs)):
				pos_idx,neg_idx = idxs[i],neg_sampled[i]
				thisPos = self.data['data'][pos_idx]
				thisNeg = self.data['data'][neg_idx] #(imgid,sent,sent_c)
				# so for training, each batch item is ((imgid,sent,sent_c),(neg_imgid,neg_sent,neg_c))
				# not using the pair with the sample image
				assert thisPos[0] != thisNeg[0]:

				data = (thisPos,thisNeg)
				out['data'].append(data)
				#out['sentids'].append(self.data['sentids'][idxs[i]])
			# replace some negative image id to decouple them?
			#print out['data'][-100:]
			"""

			# now we return 128 pos pair and 128 neg pair, pos-neg pair will be done on the fly
			out['pos'] = []
			out['neg'] = []
			# get the sample pool with no image overlap with the positives
			pos_img_ids = {self.data['data'][i][0]:1 for i in idxs}
			neg_pool = [idx for idx in self.valid_idxs if idx not in idxs and not pos_img_ids.has_key(self.data['data'][idx][0])]
			neg_sampled = random.sample(neg_pool,len(idxs)) 
			for i in xrange(len(idxs)):
				pos_idx,neg_idx = idxs[i],neg_sampled[i]
				thisPos = self.data['data'][pos_idx]
				thisNeg = self.data['data'][neg_idx] #(imgid,sent,sent_c)
				
				out['pos'].append(thisPos)
				out['neg'].append(thisNeg)



		else:
			# the idx is len(imgids)* len(sentids), so sample from the self.data_list get the indice first, then get the (imgid. sent) data
			out['imgs'] = []
			out['sentids'] = []
			imgids = {}
			for i in idxs:
				imgid,sent,sent_c = self.data['data'][i]
				imgids[imgid] = 1
				out['data'].append((sent,sent_c))
				out['sentids'].append(self.data['sentids'][i])
			out['imgs'] = imgids.keys()


		# so we get a batch_size of data : {"q":[] -> len() == batch_size}
		#assert len(out['data']) > 0
		return out


	# should return num_steps -> batches
	# step is total/batchSize * epoch
	# cap means limits max number of generated batches to 1 epoch
	def get_batches(self, batch_size, num_steps=None, shuffle=True, no_img_feat=False, full=False):
		
		num_batches_per_epoch = int(math.ceil(self.num_examples / float(batch_size)))
		# batch_size            = 256
		# num_steps             = int(math.ceil(train_data.num_examples/float(config.batch_size)))*config.num_epochs
		# num_steps             = 34920 ? 
		# shuffle               = True
		# no_img_feat           = True
		# full                  = False
		# self.num_examples     = 148915
		# num_batches_per_epoch = 512 ( ceil(581.69921875)) > 512 batch가 run 해야 1 epoch

		if full: # False, 그냥 지나감
			num_steps = num_batches_per_epoch
			
		# this may be zero
		num_epochs = int(math.ceil(num_steps/float(num_batches_per_epoch)))
		# 60 = 34920/512
		
		# shuflle
		if(shuffle): # True
			# shuffled idx
			# but all epoch has the same order
			random_idxs    = random.sample(self.valid_idxs, len(self.valid_idxs))
			# random_idxs  = [....84256, 144972, 91379, 146291, 114578, 110674, 32394, 53312, 62526, 139815]			
			# 148915       = len(random_idxs)
			
			# 582         = len(grouper(random_idxs,batch_size))
			# 581.699     = 148915/256(batch_size)		
			# step은 단지 전체 학습셋을 batch size로 등분하는것일뿐
			# 즉, 582 step(등분)을 돌리면 1 epoch란 의미 == 1 epoch 마다 582 번의 step
			
			# list(grouper(random_idxs,batch_size)
			# 	- [(256),()()()()()()()..()] 즉, 582개 ()를 가진 리스트
			# 	- 256 의 길이를 가진 실제값(14236, 128688, 106198, 1274, 91327, 60272, 125655, 146316, 88751, 53335, 650, 36521, 23977, 59944, 71741, 89851, 48614, 56190, 64523, 60290, 7380, 104885, 121954, 14054, 112705, 144979, 106900, 36632, 8875, 15757, 19025, 65254, 145896, 63975, 58569, 140905, 126687, 55771, 52536, 31218, 58188, 57586, 68725, 58431, 126382, 47408, 121660, 30032, 10306, 127682, 94917, 97994, 132740, 5536, 130659, 43638, 65523, 56349, 141139, 78198, 119676, 72159, 36883, 11532, 72095, 73042, 119036, 130308, 86963, 84735, 237, 69396, 34277, 135128, 54534, 53927, 71433, 30574, 36952, 34172, 14429, 137702, 40703, 122998, 81428, 120354, 57579, 120606, 90196, 97925, 148389, 26664, 117240, 54361, 106, 87891, 52762, 127474, 40039, 48463, 97571, 113646, 140582, 66071, 5583, 50111, 6485, 19748, 145174, 80231, 82617, 113131, 29432, 84226, 48703, 8422, 89190, 70047, 48394, 61825, 63677, 111726, 15046, 43883, 14047, 383, 86887, 145478, 103898, 62341, 71125, 55014, 53974, 140770, 106947, 128717, 72378, 87204, 51291, 148833, 76654, 93136, 47435, 144887, 16516, 119406, 77338, 30123, 28877, 126725, 80528, 75929, 19980, 120259, 73377, 78351, 62733, 142226, 129821, 73066, 140923, 79900, 61514, 96311, 101665, 65516, 5465, 13645, 88848, 50793, 55324, 7158, 48951, 90314, 108467, 41472, 81416, 36110, 139140, 27605, 54593, 49859, 74460, 141990, 121536, 61473, 100317, 126817, 42570, 19459, 129652, 76365, 70301, 80496, 3784, 82927, 118540, 89696, 88799, 136374, 93211, 44436, 144774, 93711, 116717, 12493, 1535, 45934, 37677, 137568, 29096, 7979, 116468, 106471, 74818, 11577, 99683, 40909, 95065, 61311, 80714, 146467, 11835, 38191, 40152, 44210, 60270, 142821, 107485, 142329, 42926, 92954, 48693, 129784, 76516, 67398, 657, 128208, 133597, 49195, 98944, 49797, 2095, 22717, 81385, 122470, 481, 99976, 7560, 143826, 102451, 83642, 12189, 71371, 94192, 40358)
			# 	- 이것들의 개수가 582개
			#       - ()안에 반드시 256크기이지만, 그 요소안에는 False들이 존재할 수 있음.
			random_grouped = lambda: list(grouper(random_idxs,batch_size)) # all batch idxs for one epoch			
			# grouper
			# given a list and n(batch_size), devide list into n sized chunks
			# last one will fill None
			grouped = random_grouped			
		else:
			raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
			grouped = raw_grouped

		# grouped is a list of list, each is batch_size items make up to -> total_sample

		# all batches idxs from multiple epochs
		batch_idxs_iter = itertools.chain.from_iterable(grouped() for _ in xrange(num_epochs))		
		# print "in get batches, num_steps:%s,num_epch:%s"%(num_steps,num_epochs)
		# in get batches, num_steps:34920, num_epch:60
		
		for _ in xrange(num_steps): # num_step should be batch_idxs length
			# so in the end batch, the None will not included
			batch_idxs = tuple(i for i in next(batch_idxs_iter) if i is not None) # each batch idxs
			# 256 = len(batch_idxs)
			# batch_idxs 의 실제값 예 : (103384, 255, 82216, 315, 122198, 58618, 67303, 91838, 127340, 45835, 89019, 9588, 105337, 135619, 117547, 141487, 2178, 102539, 117145, 59387, 21116, 1040, 116742, 61008, 123464, 8349, 130557, 47565, 33572, 14298, 31391, 59931, 59663, 6931, 74193, 125466, 110874, 73967, 101230, 78586, 86979, 8020, 42780, 92366, 48040, 129665, 112231, 130548, 107250, 133419, 70990, 85949, 65505, 14140, 100479, 103795, 8189, 64692, 144297, 10991, 11601, 46047, 48832, 109761, 141227, 89356, 8147, 85701, 102092, 124976, 138016, 116159, 141220, 17432, 108695, 38201, 147940, 122831, 113733, 20319, 79890, 7747, 21256, 101136, 67840, 24564, 80880, 121205, 124985, 14926, 55055, 91260, 53356, 130231, 37041, 15223, 19068, 101772, 64788, 67953, 69781, 114382, 123440, 53178, 95273, 127044, 77945, 125432, 112836, 100194, 32244, 98975, 103798, 140811, 28030, 90561, 106491, 5752, 129106, 90730, 72276, 10387, 6638, 2560, 9353, 129570, 111481, 38361, 118217, 67262, 109834, 27641, 50855, 93192, 126579, 104143, 46356, 3175, 136423, 110010, 104425, 110493, 126617, 75033, 67301, 118447, 43760, 37421, 13225, 81985, 108939, 50414, 95993, 63113, 140100, 6484, 131724, 118443, 68213, 33157, 31346, 131128, 58560, 46683, 117412, 110002, 132192, 114546, 3760, 120378, 48244, 62479, 137884, 18703, 60776, 133609, 78062, 41704, 108257, 34779, 10350, 15622, 19905, 137990, 144188, 50753, 38837, 87238, 27865, 30467, 81065, 44103, 42838, 145574, 25423, 146280, 47041, 143406, 11349, 131735, 44792, 71563, 242, 101202, 136006, 106109, 55759, 121388, 78689, 19249, 137693, 23595, 51396, 132141, 51380, 75587, 62445, 116458, 673, 135117, 142227, 147281, 78069, 106875, 40900, 45460, 40940, 108117, 147459, 60361, 131149, 41592, 42088, 96235, 142903, 56297, 9219, 109311, 17337, 3439, 23323, 77165, 18768, 94979, 137058, 60137, 26611, 65709, 80446, 4174, 15319, 127746, 25247, 127105, 130648, 17906)
			# so batch_idxs might not be size batch_size -> 
			#	- 이 의미는 grouped 안에 ()안에 반드시 256크기이지만, 그 요소안에는 False들이 존재, 256개보다 작을수 있다.
			
			# batch_data
			# 	a dict of {"data":[]} > 실제데이터
			# 	 실제로는 이런식의 데이터가 존재 : .. ('3583092048', ['two', 'female', 'performers', 'are', 'dressed', 'in', 'eccentric', 'clothing', ',', 'one', 'wearing', 'red', 'and', 'looks', 'un', 'amused', 'and', 'one', 'wearing', 'blue', 'that', 'looks', 'extremely', 'happy', '.'], [['t', 'w', 'o'], ['f', 'e', 'm', 'a', 'l', 'e'], ['p', 'e', 'r', 'f', 'o', 'r', 'm', 'e', 'r', 's'], ['a', 'r', 'e'], ['d', 'r', 'e', 's', 's', 'e', 'd'], ['i', 'n'], ['e', 'c', 'c', 'e', 'n', 't', 'r', 'i', 'c'], ['c', 'l', 'o', 't', 'h', 'i', 'n', 'g'], [','], ['o', 'n', 'e'], ['w', 'e', 'a', 'r', 'i', 'n', 'g'], ['r', 'e', 'd'], ['a', 'n', 'd'], ['l', 'o', 'o', 'k', 's'], ['u', 'n'], ['a', 'm', 'u', 's', 'e', 'd'], ['a', 'n', 'd'], ['o', 'n', 'e'], ['w', 'e', 'a', 'r', 'i', 'n', 'g'], ['b', 'l', 'u', 'e'], ['t', 'h', 'a', 't'], ['l', 'o', 'o', 'k', 's'], ['e', 'x', 't', 'r', 'e', 'm', 'e', 'l', 'y'], ['h', 'a', 'p', 'p', 'y'], ['.']])...
			# ? for training, will also sample negative images and sentences
			# 	-> get_by_idxs pos nad neg sampling
			batch_data = self.get_by_idxs(batch_idxs) # get the actual data based on idx

			if not no_img_feat:
				
				imgid2idx = {} # get all the imiage to index
				if self.is_train:
					flip = batch_data['flip']
					for pos,neg in batch_data['data']:
						imgid_pos = pos[0]
						imgid_neg = neg[0]
						if(not imgid2idx.has_key(imgid_pos)):
							imgid2idx[imgid_pos] = len(imgid2idx.keys())# start from zero
						if(not imgid2idx.has_key(imgid_neg)):
							imgid2idx[imgid_neg] = len(imgid2idx.keys())# start from zero
				else:
					"""
					for imgid,_,_ in batch_data['data']:
						if(not imgid2idx.has_key(imgid)):
							imgid2idx[imgid] = len(imgid2idx.keys()) # start from zero
					"""
					for imgid in batch_data['imgs']:
						imgid2idx[imgid] = len(imgid2idx.keys()) # img id overlap?

				# TODO: put the feature matrix construction during read_data stage, as image_emb_mat like word vectors emb_mat
				# fill in the image feature
				# imgdim is a list now
				#image_feats = np.zeros((len(imgid2idx),self.imgfeat_dim),dtype="float32")
				image_feats = np.zeros([len(imgid2idx)] + self.imgfeat_dim,dtype="float32")

				# here image_matrix idx-> feat, will replace the pid in each instance to this idx
				for imgid in imgid2idx: # fill each idx with feature, -> pid
					if self.shared['imgid2feat'] is None:
						# load feature online
						"""
						assert self.shared['featpath'] is not None
						if self.shared['featCache'].has_key(imgid):
							feat = self.shared['featCache'][imgid]
						else:
							feat = np.load(os.path.join(self.shared['featpath'],"%s.npy"%imgid))
							if len(self.shared['featCache']) <= self.shared['cacheSize']:
								self.shared['featCache'][imgid] = feat
						"""
						feat = np.load(os.path.join(self.shared['featpath'],"%s.npy"%imgid))
					else:
						feat = self.shared['imgid2feat'][imgid]				

					image_feats[imgid2idx[imgid]] = feat


				batch_data['imgidx2feat'] = image_feats

				batch_data['imgid2idx'] = imgid2idx # will used during generating feed_dict to get the 

			

			yield batch_idxs,Dataset(batch_data,self.datatype,shared=self.shared,is_train=self.is_train,imgfeat_dim=self.imgfeat_dim)






VERY_NEGATIVE_NUMBER = -1e30


# exponetial mask (so the False element doesn't get zero, it get a very_negative_number so that e(numer) == 0.0)
# [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
def exp_mask(val,mask):
	# tf.cast(a,"float") -> [True,True,False] -> [1.0,1.0,0.0] (1 - cast) -> [0.0,0.0,1.0]
	# then the 1.0 * very_negative_number and become a very_negative_number (add val and still very negative), then e(ver_negative_numer) is zero
	return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name="exp_mask")


# flatten a tensor
# [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
def flatten(tensor, keep): # keep how many dimension in the end, so final rank is keep + 1
	# get the shape
	fixed_shape = tensor.get_shape().as_list() #[N, JQ, di] # [N, M, JX, di] 
	start = len(fixed_shape) - keep # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
	# each num in the [] will a*b*c*d...
	# so [0] -> just N here for left
	# for [N, M, JX, di] , left is N*M
	left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
	# [N, JQ,di]
	# [N*M, JX, di] 
	out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
	# reshape
	flat = tf.reshape(tensor, out_shape)
	return flat

def reconstruct(tensor, ref, keep): # reverse the flatten function
	ref_shape = ref.get_shape().as_list()
	tensor_shape = tensor.get_shape().as_list()
	ref_stop = len(ref_shape) - keep
	tensor_start = len(tensor_shape) - keep
	pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
	keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
	# pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
	# keep_shape = tensor.get_shape().as_list()[-keep:]
	target_shape = pre_shape + keep_shape
	out = tf.reshape(tensor, target_shape)
	return out

def evaluate(dataset,config,sess,tester):
	imgid2vec = {}
	sentid2vec = {}

	#num_steps = int(math.ceil(dataset.num_examples/float(config.batch_size)))
	for evalbatch in dataset.get_batches(config.batch_size,shuffle=False,full=True):
		# each batch is separte images and sentences
		z_u,z_v = tester.step(sess,evalbatch) # [N,hop+1,d]
		#print z_u.shape
		
		for sentId,i in zip(evalbatch[1].data['sentids'],range(len(z_u))):
			sentid2vec[sentId] = z_u[i]
		for imgId,i in zip(evalbatch[1].data['imgs'],range(len(z_v))):
			imgid2vec[imgId] = z_v[i]
	
	tqdm.write("total image:%s, sent:%s"%(len(imgid2vec),len(sentid2vec)))
	# now getting similarity score
	imgid2sents = {}
	sentId2imgs = {}
	for imgid in tqdm(imgid2vec.keys(),ascii=True):
		for sentid in sentid2vec:
			imgvec = imgid2vec[imgid] # [hop+1,d]
			sentvec = sentid2vec[sentid]
			#print imgid,imgvec
			#print sentid,sentvec

			s = np.sum(imgvec*sentvec)
			#print s
			#sys.exit()

			if not imgid2sents.has_key(imgid):
				imgid2sents[imgid] = []
			imgid2sents[imgid].append((sentid,s))
			# sent to img
			if not sentId2imgs.has_key(sentid):
				sentId2imgs[sentid] = []
			sentId2imgs[sentid].append((imgid,s))
	
	return getEvalScore(imgid2sents,sentId2imgs)

# recall @1,5,10,median rank of the top-ranked ground truth
def getEvalScore(imgid2sents,sentid2imgs):
	# img to text
	i2t = {"r@1":[],"r@5":[],'r@10':[],'mr':[]}
	for imgId in imgid2sents:
		imgid2sents[imgId].sort(key=operator.itemgetter(1),reverse=True)
		#print imgid2sents[imgId][:30]
		#sys.exit()
		# find the first ground truth sent for this image
		gtrank=-1
		for i,(sentId,score) in enumerate(imgid2sents[imgId]):
			thisimgId,_ = sentId.strip().split("#")
			thisimgId = os.path.splitext(thisimgId)[0]
			if thisimgId == imgId:
				gtrank = i+1
				break
		assert gtrank>0

		if gtrank <= 1:
			i2t["r@1"].append(1)
		else:
			i2t["r@1"].append(0)

		if gtrank <= 5:
			i2t["r@5"].append(1)
		else:
			i2t["r@5"].append(0)

		if gtrank <= 10:
			i2t["r@10"].append(1)
		else:
			i2t["r@10"].append(0)

		i2t['mr'].append(gtrank)


	p = {
		"i2t_r@1":sum(i2t["r@1"])/float(len(i2t["r@1"])),
		"i2t_r@5":sum(i2t["r@5"])/float(len(i2t["r@5"])),
		'i2t_r@10':sum(i2t["r@10"])/float(len(i2t["r@10"])),
		'i2t_mr':np.median(i2t['mr'])
	}

	# text to img
	t2i = {"r@1":[],"r@5":[],'r@10':[],'mr':[]}
	for sentId in sentid2imgs:
		sentid2imgs[sentId].sort(key=operator.itemgetter(1),reverse=True)
		thisimgId,_ = sentId.strip().split("#")
		thisimgId = os.path.splitext(thisimgId)[0]
		# find the first ground truth sent for this image
		gtrank=-1
		for i,(imgId,score) in enumerate(sentid2imgs[sentId]):
			
			if thisimgId == imgId:
				gtrank = i+1
				break
		assert gtrank>0

		if gtrank <= 1:
			t2i["r@1"].append(1)
		else:
			t2i["r@1"].append(0)

		if gtrank <= 5:
			t2i["r@5"].append(1)
		else:
			t2i["r@5"].append(0)

		if gtrank <= 10:
			t2i["r@10"].append(1)
		else:
			t2i["r@10"].append(0)

		t2i['mr'].append(gtrank)


	p.update({
		"t2i_r@1":sum(t2i["r@1"])/float(len(t2i["r@1"])),
		"t2i_r@5":sum(t2i["r@5"])/float(len(t2i["r@5"])),
		't2i_r@10':sum(t2i["r@10"])/float(len(t2i["r@10"])),
		't2i_mr':np.median(t2i['mr'])
	})
		
	return p

"""
max_num_albums:8 ,max_num_photos:10 ,max_sent_title_size:35 ,max_sent_des_size:2574 ,max_when_size:4 ,max_where_size:10 ,max_answer_size:18 ,max_question_size:25 ,max_word_size:42
"""

# datasets[0] should always be the train set
def update_config(config,datasets,showMeta=False):
	config.max_sent_size = 0
	config.max_word_size = 0 # word letter count

	# go through all datasets to get the max count
	for dataset in datasets:
		for idx in xrange(len(dataset.data['data'])):# dataset.valid_idxs:

			imageId,sent,sent_c = dataset.data['data'][idx]
			
			config.max_sent_size = max(config.max_sent_size,len(sent))

			config.max_word_size = max(config.max_word_size, max(len(word) for word in sent))

	
	if showMeta:
		config_vars = vars(config)
		print "max meta:"
		print "\t" + " ,".join(["%s:%s"%(key,config_vars[key]) for key in config.maxmeta])

	# adjust the max based on the threshold argument input as well
	if config.is_train:
		config.max_sent_size = min(config.max_sent_size,config.sent_size_thres)
				

	# always clip word_size
	config.max_word_size = min(config.max_word_size, config.word_size_thres)

	# get the vocab size # the charater in the charCounter
	config.char_vocab_size = len(datasets[0].shared['char2idx'])
	# the word embeding's dimension
	# so just get the first vector to see the dim
	if not config.no_wordvec:
		config.word_emb_size = len(next(iter(datasets[0].shared['word2vec'].values())))

	#config.imgfeat_size = len(next(iter(datasets[0].shared['imgid2feat'].values())))
	# the size of word vocab not in existing glove, if finetining should be differenet
	config.word_vocab_size = len(datasets[0].shared['word2idx'])
