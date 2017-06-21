import nltk, os
from nltk.parse import stanford
import gensim
import config
import tensorflow as tf
os.environ['STANFORD_PARSER'] = "/home/manpreet/Desktop/stanford-parser/stanford-parser.jar"
os.environ['STANFORD_MODELS'] = "/home/manpreet/Desktop/stanford-parser/stanford-parser-3.7.0-models.jar"

def build_dataset(words):
	return vocab_index

#def create_input(sentence):
#	dep_parser = stanford.StanfordDependencyParser(model_path="/home/manpreet/Desktop/stanford-parser/englishPCFG.ser.gz")
#	dep_parse = [list(parse.triples()) for parse in dep_parser.raw_parse(sentence)]

def create_input(sentence, word_to_id):
	tokens = [token for token in sentence.split()]
	input = [word_to_id[token] if token in word_to_id else 'UNK' for token in tokens]
	return input


def word_to_index(vocab):
	"""
	Maps words to their index in embedding file.
	"""
	pretrained_loc = config.PRETRAINED_WORDS
	with open(pretrained_loc, 'r') as f:
		words = f.read().split('\n')
	dico = {}
	index = {}
	i = 0
	for word in vocab:
		if word in words:
			dico[word] = words.index(word)
			index[word] = i
			i+=1
	index['UNK'] = i
	index['PAD'] = i+1
	return dico, index



def convert_to_id(input_x, word_to_id, batch_size):
	seqlen = []
	maxseqlen = 0
	for item in input_x:
		if len(item)>maxseqlen:
			maxseqlen = len(item)
	Idx = []
	for sent in input_x:
		sentIdx = []
		for word in sent:
			try:
				sentIdx.append(word_to_id[word])
			except KeyError:
				sentIdx.append(word_to_id['UNK'])
		seqlen.append(len(sent))
		if seqlen[-1]<maxseqlen:
			sentIdx += [word_to_id['PAD']]*(maxseqlen-seqlen[-1])
		Idx.append(sentIdx)
	return maxlenseq, seqlen, Idx


def convert_tag_to_id(tags_list, maxlenseq):
	tag_to_id = {}
	tag_to_id['PAD'] = 0
	i = 1
	Idx = []
	for sent in tags:
		sentIdx = []
		for word in sent:
			if tag not in tag_dict:
				tag_to_id[tag] = i
				i+=1
			sentIdx.append(tag_to_id[tag])
		if len(sent)<maxseqlen:
			sentIdx += [[tag_to_id['PAD']]*(maxseqlen-len(sent))
		Idx.append(sentIdx)
	return Idx, tag_to_id


def create_batches(input_x, input_y, seqlen, batch_size, maxlenseq):
	#inp = np.concatenate((input_x, input_y), axis=1)
	for num, item in enumerate(input_x):
		inp.append([item, input_y[num]])
	inp = sorted(inp, key=dict(zip(inp, seqlen)).get)
	batch_len = len(input_x)//batch_size
	inp = tf.convert_to_tensor(batch_x, name="input_data_x", dtype=tf.int32)
	inp = pad_last_batch(inp, batch_size)
	input_x = tf.reshape(inp, [batch_len+1, batch_size])
	return batches


def pad_lastbatch(inp, batch_size, maxlenseq):
	num_unbatched = len(data)%batch_size
	if num_unbatched!=0:
		data+=[[0]*maxseqlen]*(batch_size-num_unbatched)
	return data




#if __name__=="__main__":

