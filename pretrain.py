import tensorflow as tf
import numpy as np
from loader import build_vocab, load_emb
from utils import word_to_index, word_to_index_glove, word_to_index_word2vec, eval
import loader, utils, config
import optparse
#from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import json


class Embedding:
    """
    Embedding class that loads the pretrained word2vec embeddings from
    gensim model or weights file into tensorflow variable format.
    """

    def __init__(self, opts, word2vec_emb_path=None, glove_emb_path=None):

        if opts.restore:
            self.weights = tf.Variable(np.ones((39762, 600)), trainable=False, name="pretrained_embeddings", dtype=tf.float32)
            with open("word_to_id", "r") as f:
                self.word_to_id = json.load(f)
            return

        train_vocab = build_vocab()

        if opts.word2vec:
            binary = word2vec_emb_path.endswith('.bin')
            word2vec_model = Word2Vec.load_word2vec_format(word2vec_emb_path, binary=binary)

        if opts.glove and opts.word2vec:
            self.weights, self.word_to_id = word_to_index(train_vocab, word2vec_model, glove_emb_path)
            del (word2vec_model)
        elif opts.glove and not opts.word2vec:
            self.weights, self.word_to_id = word_to_index_glove(train_vocab, glove_emb_path)
        else:
            self.weights, self.word_to_id = word_to_index_word2vec(train_vocab, word2vec_model)
            # TODO:Better way to do Memory Management
            del(word2vec_model)
        self.weights.astype(np.float32)
        self.weights = tf.Variable(self.weights, trainable=False, name="pretrained_embeddings", dtype=tf.float32)

    def lookup(self, sentences):
        return tf.nn.embedding_lookup(self.weights, sentences)


class BLSTM:
    def __init__(self, lstm_size):
        ##No need for max num of time steps
        self.lstm_size = lstm_size
        self.cell_fw = tf.contrib.rnn.LSTMCell(num_units=lstm_size, state_is_tuple=True)
        self.cell_bw = tf.contrib.rnn.LSTMCell(num_units=lstm_size, state_is_tuple=True)

    def forward(self, input, input_length, var_name):
        if config.keep_prob < 1:
            input = tf.nn.dropout(input, config.keep_prob)
        with tf.variable_scope(var_name):
            output, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                                                 dtype=tf.float32, sequence_length=input_length,
                                                                 inputs=input)
            output = tf.concat(output, 2)
            last_state = tf.concat(last_state, 2)
        return output, last_state[1]


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
        inp = tf.reshape(inputs, [-1, lstm_size])
        logits = tf.add(tf.matmul(inp, self.weights), self.biases)
        num_labels = int(logits.get_shape()[1])
        logits = tf.reshape(logits, [config.batch_size, -1, num_labels])
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

    optparser = optparse.OptionParser()
    optparser.add_option(
        "-g", "--glove", default=True,
        help="Use glove embeddings"
    )
    optparser.add_option(
        "-c", "--crf", default=True,
        help="Use CRF"
    )
    optparser.add_option(
        "-w", "--word2vec", default=True,
        help="Use word2vec embeddings"
    )
    optparser.add_option(
        "-e", "--char_emb", default=False,
        help="Run character-level embeddings"
    )
    optparser.add_option(
        "-r", "--restore", default=True,
        help="Rebuild the model and restore weights from checkpoint"
    )
    opts = optparser.parse_args()[0]

    batch_size = config.batch_size
    word2vec_emb_path = config.word2vec_emb_path
    glove_emb_path = config.glove_emb_path
    input_x, input_y = loader.prepare_input(config.datadir+config.train)
    if opts.char:
        char_emb, char_to_id, char_seq_len = utils.convert_to_char_emb(input_x)
        char_layer = BLSTM(config.char_lstm_size)
    emb_layer = Embedding(opts, word2vec_emb_path, glove_emb_path)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
    seqlen, inp = utils.create_batches(input_x, seqlen, input_y)

    num_labels = len(tag_to_id)
    lstm_size = 100
    blstm_layer = BLSTM(lstm_size)
    ff_layer = FeedForward(2*config.lstm_size, num_labels)

    if opts.char:
        #dimension of batch and sequence_len are collapsed as batch_size is 1.
        char_inp = tf.placeholder("float32", shape=[None, None, len(char_to_id)], name="char_input")
        char_seqlen = tf.placeholder("int32", shape=[None], name="char_seqlen")
    batch_input = tf.placeholder("int32", shape=[None, None], name="input")
    sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
    if opts.crf:
        labels = tf.placeholder("int32", shape=[None, None],  name="labels")
    else:
        labels = tf.placeholder("int32", shape=[None, None, num_labels],  name="labels")

    #loss_mask = tf.placeholder("float64", shape=[None])
    word_embeddings = emb_layer.lookup(batch_input)
    word_embeddings = tf.cast(word_embeddings, tf.float32)
    if opts.char:
        _, char_embeddings = char_layer.forward(char_inp, char_seqlen, "CharLSTM1")
        char_embeddings = tf.expand_dims(char_embeddings, 0)
        embeddings = tf.concat([char_embeddings, word_embeddings], 2)
    else:
        embeddings = word_embeddings
    hidden_output, _ = blstm_layer.forward(embeddings, sequence_length, "SourceLSTM")
    unary_potentials = ff_layer.forward(hidden_output)
    #unary_potentials = tf.reshape(unary_potentials, [config.batch_size, -1, num_labels])
    if opts.crf:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_potentials, labels, sequence_length)
        cost =  tf.reduce_mean(-log_likelihood)
    else:
        cost = loss(unary_potentials, labels)
    train_op = train(cost)

    sess = tf.Session()
    if opts.restore:
        saver = tf.train.Saver()
        saver.restore(sess, "./source_model_crf")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    batch_len = len(inp)//batch_size
    loss2 = []

    loss_ = []
    for _ in range(config.num_epochs):
        loss_.append([])
        for seq_len, batch in zip(seqlen, inp):
            x = []
            y = []
            for b in batch:
                x.append(b[0])
                tags = b[1]
                y.append([])
                for label in tags:
                    if opts.crf:
                        y[-1].append(label)
                    else:
                        tag = [0]*num_labels
                        tag[label] = 1
                        y[-1].append(tag)
            sess.run(train_op, feed_dict={batch_input:x, labels:y, sequence_length:seq_len})
            loss_[-1].append(sess.run(cost, feed_dict={batch_input:x, labels:y, sequence_length:seq_len}))
            print loss_[-1][-1]

    loader.save_smodel(sess)

    ##Run model on test data
    input_x, input_y = loader.prepare_input(config.datadir + config.test)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y = utils.convert_tag_to_id(tag_to_id, input_y)
    seqlen, inp = utils.create_batches(input_x, seqlen, input_y)

    ##Run on Medpost (target) data
    input_x, input_y = loader.prepare_medpost_input()
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y = utils.convert_tag_to_id(tag_to_id, input_y)
    seqlen, inp = utils.create_batches(input_x, seqlen, input_y)

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
                if opts.crf:
                    y[-1].append(label)
                else:
                    tag = [0] * num_labels
                    tag[label] = 1
                    y[-1].append(tag)
        if opts.crf:
            trans_mat = sess.run(transition_params)
            unary_pot = sess.run(unary_potentials, feed_dict={batch_input: x, labels: y, sequence_length: seq_len})
            #CRF decodes only one sequence at a time
            #TODO: Decoding only 1st sequence as batch_size is 1. Change if batch_size increases.
            pred, _ = tf.contrib.crf.viterbi_decode(unary_pot[0], trans_mat)
        else:
            pred = sess.run(unary_potentials, feed_dict={batch_input: x, labels: y, sequence_length: seq_len})
        #Only the first sequence since batch_size=1
        if opts.crf:
            for t, p in zip(y[0], pred):
                print "Predicted ", p
                print "True ", t
                predictions.append(p)
                true_labels.append(t)
        else:
            #TODO: Change design for batch_size>1.
            for t, p in zip(y[0], pred[0]):
                print "Predicted ", np.argmax(p)
                print "True ", np.argmax(t)
                predictions.append(np.argmax(p))
                true_labels.append(np.argmax(t))

    eval(predictions, true_labels, tag_to_id)

