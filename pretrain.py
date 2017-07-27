import tensorflow as tf
import numpy as np
from loader import build_vocab, load_emb
from utils import word_to_index, eval
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
            pad = np.zeros(shape=self.emb_size, dtype='float32')
            unk = np.random.normal(0.001, 0.01, self.emb_size)
            weights = np.vstack((weights, unk, pad))
            self.weights = tf.Variable(weights, trainable=False, name="pretrained_embeddings")
            self.weights.dtype = np.float32
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
        with tf.variable_scope("SourceLSTM"):
            output, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                                                 dtype=tf.float32, sequence_length=input_length,
                                                                 inputs=input)
            output = tf.concat(output, 2)
        return output


class FeedForward:
    def __init__(self, input_size, num_labels):
        # get_variable because softmax_w and softmax_b will be called multiple times during training.
        self.weights = tf.Variable(tf.random_normal([input_size, num_labels], stddev=0.035, dtype=tf.float32),
                                   name="weights", trainable=True)
        self.biases = tf.Variable(tf.zeros(num_labels, dtype=tf.float32), name="biases", trainable=True)

    def forward(self, inputs):
        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        # inp is reshaped to (-1, v_dim)
        # and multiplication happens for all examples in the batch
        # Output logits is of the form [num_labels, batch_size*sequence_length]
        lstm_size = int(inputs.get_shape()[2])
        inp = tf.reshape(tf.stack(axis=0, values=inputs), [-1, lstm_size])
        print inp.get_shape()
        logits = tf.add(tf.matmul(inp, self.weights), self.biases)
        return logits


def loss(logits, labels, mask=None):
    """docstring for CrossEntropy"""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #cost = tf.reduce_sum(tf.multiply(loss, mask))
    cost = tf.reduce_mean(loss)
    return cost


def train(cost):
    _lr = tf.Variable(0.005, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(_lr)
    _train_op = optimizer.minimize(cost)
    return _train_op


def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


if __name__ == "__main__":

    batch_size = config.batch_size
    ext_emb_path = config.ext_emb_path
    input_x, input_y = loader.prepare_input(config.datadir+config.train)
    emb_layer = Embedding(ext_emb_path)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
    seqlen, inp = utils.create_batches(input_x, input_y, seqlen, config.batch_size)

    num_labels = len(tag_to_id)
    lstm_size = 100
    blstm_layer = BLSTM(lstm_size)
    ff_layer = FeedForward(2*lstm_size, num_labels)

    batch_input = tf.placeholder("int32", shape=[None, None], name="input")
    sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
    labels = tf.placeholder("int32", shape=[None, None],  name="labels")
    #loss_mask = tf.placeholder("float64", shape=[None])
    embeddings = emb_layer.lookup(batch_input)
    hidden_output = blstm_layer.forward(embeddings, sequence_length)
    unary_potentials = ff_layer.forward(hidden_output)
    unary_potentials = tf.reshape(unary_potentials, [config.batch_size, -1, num_labels])
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_potentials, labels, sequence_length)
    loss =  tf.reduce_mean(-log_likelihood)
    #cost = loss(logits, labels)
    train_op = train(loss)

    sess = tf.Session()
    if config.restore:
        saver = tf.train.Saver()
        saver.restore(sess, "./source_model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    batch_len = len(inp)//batch_size
    loss2 = []

    loss = []
    for _ in range(config.num_epochs):
        loss.append([])
        for seq_len, batch in zip(seqlen, inp):
            x = []
            y = []
            for b in batch:
                x.append(b[0])
                tags = b[1]
                y.append([])
                for label in tags:
                    y[-1].append(label)
            sess.run(train_op, feed_dict={batch_input:x, labels:y, sequence_length:seq_len})
            loss[-1].append(sess.run(loss, feed_dict={batch_input:x, labels:y, sequence_length:seq_len}))
            print loss[-1][-1]

    loader.save_smodel(sess)

    ##Run model on test data
    input_x, input_y = loader.prepare_input(config.datadir + config.test)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y = utils.convert_tag_to_id(tag_to_id, input_y)
    seqlen, inp = utils.create_batches(input_x, input_y, seqlen, config.batch_size)
    predictions = []
    true_labels = []
    for seq_len, batch in zip(seqlen, inp):
        x = []
        y = []
        for b in batch:
            x.append(b[0])
            tags = b[1]
            y.append([])
            for label in tags:
                tag = [0] * num_labels
                tag[label] = 1
                y[-1].append(tag)
        unary_pot, trans_mat = sess.run(unary_potentials, transition_params, feed_dict={batch_input: x, labels: y, sequence_length: seq_len})
        pred, _ = tf.contrib.crf.viterbi_decode(unary_pot, trans_mat)
        for t, p in zip(y, pred):
            print "Predicted ", np.argmax(p)
            print "True ", np.argmax(t)
            predictions.append(np.argmax(p))
            true_labels.append(np.argmax(t))

    eval(predictions, true_labels, tag_to_id)


