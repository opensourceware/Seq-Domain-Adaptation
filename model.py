import tensorflow as tf
import numpy as np
from loader import build_vocab, load_emb
from utils import word_to_index
import loader, utils, config
import optparse
#from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec


class Embedding:
    """
    Embedding class that loads the pretrained word2vec embeddings from
    gensim model or weights file into tensorflow variable format.
    """

    def __init__(self, ext_emb_path, vocab_path=None):
        if vocab_path is None:
            binary = ext_emb_path.endswith('.bin')
            #model = KeyedVectors.load_word2vec_format(ext_emb_path, binary=binary)
            model = Word2Vec.load_word2vec_format(ext_emb_path, binary=binary)
            train_vocab = build_vocab()
            self.weights, self.word_to_id = word_to_index(train_vocab, model)
            self.emb_size = model['the'].shape[0]
            self.voc_size = len(self.word_to_id)
            #TODO:Better way to do Memory Management
            del(model)
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
        # self.weights = tf.stack([weights, pad_zeros])

    def lookup(self, sentences):
        return tf.nn.embedding_lookup(self.weights, sentences)


class BLSTM:
    def __init__(self, lstm_size):
        ##No need for max num of time steps
        self.lstm_size = lstm_size
        self.cell_fw = tf.contrib.rnn.LSTMCell(num_units=lstm_size, state_is_tuple=True)
        self.cell_bw = tf.contrib.rnn.LSTMCell(num_units=lstm_size, state_is_tuple=True)

    def forward(self, input, input_length):
        if config.keep_prob < 1:
            input = tf.nn.dropout(input, config.keep_prob)
        output, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                                             dtype=tf.float64, sequence_length=input_length,
                                                             inputs=input)
        output = tf.concat(output, 2)
        return output


class FeedForward:
    def __init__(self, input_size, num_labels):
        # get_variable because softmax_w and softmax_b will be called multiple times during training.
        self.weights = tf.Variable(tf.random_normal([input_size, num_labels], stddev=0.035, dtype=tf.float64),
                                   name="weights", trainable=True)
        self.biases = tf.Variable(tf.zeros(num_labels, dtype=tf.float64), name="biases", trainable=True)

    def forward(self, inputs):
        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        # inp is reshaped to (-1, v_dim)
        # and multiplication happens for all examples in the batch
        # Output logits is of the form [batch_size*sequence_length, num_labels]
        lstm_size = int(inputs.get_shape()[2])
        inp = tf.reshape(tf.stack(axis=0, values=inputs), [-1, lstm_size])
        logits = tf.transpose(tf.add(tf.matmul(inp, self.weights), self.biases))
        return logits


class FeedForwardTrg:
    def __init__(self, input_size, num_labels):
        # get_variable because softmax_w and softmax_b will be called multiple times during training.
        self.weights = tf.Variable(tf.random_normal([input_size, num_labels], stddev=0.035, dtype=tf.float64),
                                   name="weights", trainable=True)
        self.biases = tf.Variable(tf.zeros(num_labels, dtype=tf.float64), name="biases", trainable=True)

    def forward(self, inputs):
        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        # inp is reshaped to (-1, v_dim)
        # and multiplication happens for all examples in the batch
        # Output logits is of the form [batch_size*sequence_length, num_labels]
        lstm_size = int(inputs.get_shape()[2])
        inp = tf.reshape(tf.stack(axis=0, values=inputs), [-1, lstm_size])
        logits = tf.transpose(tf.add(tf.matmul(inp, self.weights), self.biases))
        return logits


def loss(logits, labels, mask=None):
    """docstring for CrossEntropy"""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #cost = tf.reduce_sum(tf.multiply(loss, mask))
    cost = tf.reduce_sum(loss)
    ##TODO: Take the average of cost.
    return cost


def train(cost):
    _lr = tf.Variable(0.3, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(_lr)
    _train_op = optimizer.apply_gradients(zip(grads, tvars))
    # _new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    # _lr_update = tf.assign(self._lr, self._new_lr)
    return _train_op


def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-e", "--embed", default="vectors.txt", help="Embedding file location")
    optparser.add_option("-v", "--vocab", default="types.txt", help="Vocab file location")
    optparser.add_option("-l", "--lstm_size", default="100", type="int", help="LSTM hidden dimension")
    optparser.add_option("-m", "--mem_size", default="100", type="int", help="LSTM hidden dimension")
    opts = optparser.parse_args()[0]
    ext_emb_path = config.ext_emb_path
    vocab_path = opts.vocab
    lstm_size = opts.lstm_size
    mem_size = opts.mem_size
    batch_size = 10
    #Data processing
    emb_layer = Embedding(ext_emb_path, vocab_path)
    input_x, input_y = loader.prepare_input(config.datadir + config.train)
    maxseqlen, seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y, tag_to_id = utils.convert_tag_to_id(input_y, maxseqlen)
    num_labels = len(tag_to_id)
    batches, batchseqlen = utils.create_batches(input_x, input_y, seqlen, batch_size, maxseqlen)

    ##TODO: Debug the following code
    blstm_layer = BLSTM(lstm_size)
    ff_layer = FeedForward(2*lstm_size, len(tag_to_id))

    batch_input = tf.placeholder("int32", shape=[None, None])
    sequence_length = tf.placeholder("int32", shape=[None])
    labels = tf.placeholder("int32", shape=[None, None])
    loss_mask = tf.placeholder("float64", shape=[None])
    embeddings = emb_layer.lookup(batch_input)
    hidden_output = blstm_layer.forward(embeddings, sequence_length)
    logits = ff_layer.forward(hidden_output)
    cost = loss(logits, labels)
    train_op = train(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


