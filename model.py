import tensorflow as tf
import numpy as np
from loader import build_vocab, load_emb
import loader, utils, config
import optparse


class Embedding:
	"""
	Embedding class that loads the pretrained word2vec embeddings from 
	gensim model or weights file into tensorflow variable format. 
	"""
	def __init__(self, ext_emb_path, vocab_path=None):
		if vocab_path is None:
			binary = ext_emb_path.endswith('.bin')
			model = KeyedVectors.load_word2vec_format(ext_emb_path, binary=binary)
			self.word_to_id = create_hash(model.vocab.keys())
			self.emb_size = model['the'].shape[0]
			self.voc_size = len(word_to_id)
			unk = np.random.normal(0.001, 0.01, self.emb_size)
			pad = np.zeros(shape=self.emb_size)
			weights = np.vstack((model.syn0, unk, pad))
		else:
			vocab = build_vocab()
			_, self.word_to_id = utils.word_to_index(vocab)
			self.voc_size = len(self.word_to_id)
			weights = np.loadtxt('emb.mat')
			self.emb_size = weights[0].shape[0]
			pad = np.zeros(shape=self.emb_size, dtype='float64')
			unk = np.random.normal(0.001, 0.01, self.emb_size)
			weights = np.vstack((weights, unk, pad))
			self.weights = tf.Variable(weights, trainable=False, name="pretrained_embeddings")
			#self.weights = tf.stack([weights, pad_zeros])

	def lookup(self, sentences):
		return tf.nn.embedding_lookup(self.weights, sentences)

def train():
	result = tf.contrib.learn.run_n(
	    {"output_fw": output_fw, "output_fb": output_fb,
	    "states_fw": states_fw, "states_bw": states_bw}, n=1, feed_dict=None)

class BLSTM:
	def __init__(self, lstm_size):
		##No need for max num of time steps
		self.lstm_size = lstm_size
		self.cell_fw = tf.contrib.rnn.LSTMCell(num_units=lstm_size, state_is_tuple=True)
		self.cell_bw = tf.contrib.rnn.LSTMCell(num_units=lstm_size, state_is_tuple=True)

	def forward(self, input, input_length):
		if config.keep_prob < 1:
			input = tf.nn.dropout(input, config.keep_prob)
		output, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw, dtype=tf.float64, sequence_length=input_length, inputs=input)
		output = tf.concat(output, 2)
		return output


class FeedForwardArg:
	def __init__(self, input_size, num_labels):
	#get_variable because softmax_w and softmax_b will be called multiple times during training.
		self.weights = tf.Variable(tf.random_normal([input_size, num_labels], stddev=0.035, dtype=tf.float64), name="weights", trainable=True)
		self.biases = tf.Variable(tf.zeros(num_labels, dtype=tf.float64), name="biases", trainable=True)

	def forward(self, inputs):
		if config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)
		#inp is reshaped to (-1, v_dim)
		#and multiplication happens for all examples in the batch
		#Output logits is of the form [batch_size*sequence_length, num_labels]
		lstm_size = int(inputs.get_shape()[2])
		inp = tf.reshape(tf.stack(axis=0, values=inputs), [-1, lstm_size])
		logits = tf.transpose(tf.add(tf.matmul(inp, self.weights), self.biases))
		return logits

class FeedForwardTrg:
	def __init__(self, input_size, num_labels):
	#get_variable because softmax_w and softmax_b will be called multiple times during training.
		self.weights = tf.Variable(tf.random_normal([input_size, num_labels], stddev=0.035, dtype=tf.float64), name="weights", trainable=True)
		self.biases = tf.Variable(tf.zeros(num_labels, dtype=tf.float64), name="biases", trainable=True)

	def forward(self, inputs):
		if config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)
		#inp is reshaped to (-1, v_dim)
		#and multiplication happens for all examples in the batch
		#Output logits is of the form [batch_size*sequence_length, num_labels]
		lstm_size = int(inputs.get_shape()[2])
		inp = tf.reshape(tf.stack(axis=0, values=inputs), [-1, lstm_size])
		logits = tf.transpose(tf.add(tf.matmul(inp, self.weights), self.biases))
		return logits


def loss(logits, labels, mask):
	"""docstring for CrossEntropy"""
	loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
	cost = tf.reduce_sum(tf.multiply(loss, mask))
	##TODO: Take the average of cost.
	return cost

def train(cost):
	_lr = tf.Variable(0.3, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
	optimizer = tf.train.GradientDescentOptimizer(_lr)
	_train_op = optimizer.apply_gradients(zip(grads, tvars))
	#_new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
	#_lr_update = tf.assign(self._lr, self._new_lr)
	return _train_op


def assign_lr(self, session, lr_value):
	session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("-e", "--embed", default="vectors.txt", help="Embedding file location")
	optparser.add_option("-v", "--vocab", default="types.txt", help="Vocab file location")
	optparser.add_option("-l", "--lstm_size", default="100", type = "int", help="LSTM hidden dimension")
	optparser.add_option("-m", "--mem_size", default="100", type = "int", help="LSTM hidden dimension")
	opts = optparser.parse_args()[0]
	ext_emb_path = opts.embed
	vocab_path = opts.vocab
	lstm_size = opts.lstm_size
	mem_size = opts.mem_size
	emb_layer = cnn.Embedding(ext_emb_path, vocab_path)
	input_x, input_y = loader.prepare_input(config.datadir+config.train)
	maxlenseq, seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
	input_y, tag_to_id = utils.convert_tag_to_id(input_y, maxlenseq)
	batches = utils.create_batches(input_x, input_y, seqlen, batch_size, maxlenseq)

	num_labels = len(tag_to_id)


#Experiment
"""


input_x, input_y = loader.prepare_input(config.datadir+config.train)
maxlenseq, seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
input_y, tag_to_id = utils.convert_tag_to_id(input_y, maxlenseq)
batches = utils.create_batches(input_x, input_y, seqlen, batch_size, maxlenseq)

"""
