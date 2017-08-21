import tensorflow as tf
import numpy as np
from loader import build_vocab, load_emb
from utils import word_to_index, word_to_index_glove, word_to_index_word2vec, eval
import loader, utils, config
import optparse
#from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import json
from lstm_mapper import SourceLSTM, TargetLSTM


class Embedding:
    """
    Embedding class that loads the pretrained word2vec embeddings from
    gensim model or weights file into tensorflow variable format.
    """

    def __init__(self, opts, word2vec_emb_path=None, glove_emb_path=None):

        if opts.restore:
            self.weights = tf.Variable(np.ones((39762, 600)), trainable=False, name="embeddings", dtype=tf.float32)
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
        self.weights = tf.Variable(self.weights, trainable=False, name="embeddings", dtype=tf.float32)

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


def initialize_embeddings(graph):
    pretrained_embeddings = graph.get_tensor_by_name("pretrained_embeddings:0")
    with tf.variable_scope("word", reuse=True):
        embeddings = tf.get_variable("embeddings:0")
        sess.run(tf.assign(embeddings, pretrained_embeddings))


class BasePOSTagger(object):

    def __init__(self, graph, opts):
        """
        Load all network's weights and biases from the pre-computed graph.
        :param graph: Pre-trained graph loaded from checkpoint.
        :param opts: Contains config of network architecture
        """
        self.pretrain_lstm_fw_weights = graph.get_tensor_by_name("SourceLSTM/bidirectional_rnn/fw/lstm_cell/weights:0")
        self.pretrain_lstm_fw_biases = graph.get_tensor_by_name("SourceLSTM/bidirectional_rnn/fw/lstm_cell/biases:0")
        self.pretrain_lstm_bw_weights = graph.get_tensor_by_name("SourceLSTM/bidirectional_rnn/bw/lstm_cell/weights:0")
        self.pretrain_lstm_bw_biases = graph.get_tensor_by_name("SourceLSTM/bidirectional_rnn/bw/lstm_cell/biases:0")
        self.classifier_weights = graph.get_tensor_by_name("weights:0")
        self.classifier_biases = graph.get_tensor_by_name("biases:0")
        if opts.crf:
            self.transitions = graph.get_tensor_by_name("transitions:0")

    def _initialize(self, sess, opts):
        """
        All BLSTM weights and biases of the child class are initialized
        to values loaded from the graph in init.
        """
        with tf.variable_scope("bidirectional_rnn"):
            with tf.variable_scope("fw"):
                with tf.variable_scope("lstm_cell"):
                    lstm_fw_weights = tf.get_variable("weights", dtype="float32")
                    lstm_fw_biases = tf.get_variable("biases", dtype="float32")
                    sess.run(tf.assign(lstm_fw_weights, self.pretrain_lstm_fw_weights))
                    sess.run(tf.assign(lstm_fw_biases, self.pretrain_lstm_fw_biases))
            with tf.variable_scope("bw"):
                with tf.variable_scope("lstm_cell"):
                    lstm_bw_weights = tf.get_variable("weights", dtype="float32")
                    lstm_bw_biases = tf.get_variable("biases", dtype="float32")
                    sess.run(tf.assign(lstm_bw_weights, self.pretrain_lstm_bw_weights))
                    sess.run(tf.assign(lstm_bw_biases, self.pretrain_lstm_bw_biases))
        ff_weights = tf.get_variable("weights:0")
        sess.run(tf.assign(ff_weights, self.classifier_weights))
        ff_biases = tf.get_variable("pretrained_embeddings:0")
        sess.run(tf.assign(ff_biases, self.classifier_biases))
        if opts.crf:
            crf_transitions = tf.get_variable("transitions:0")
            sess.run(tf.assign(crf_transitions, self.transitions))


class POSTagger(BasePOSTagger):
    def __init__(self, graph, sess, opts, placeholders, emb_layer, scope):
        super(POSTagger, self).__init__(graph, opts)
        input_x, input_y = loader.prepare_input(config.datadir+config.train)
        input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
        self.placeholders = placeholders
        self.num_labels = len(tag_to_id)

        self.emb_layer = emb_layer
        with tf.variable_scope(scope):
            self.blstm_layer = BLSTM(config.lstm_size)
            self.ff_layer = FeedForward(2*config.lstm_size, self.num_labels)

            #loss_mask = tf.placeholder("float64", shape=[None])
            word_embeddings = self.emb_layer.lookup(placeholders['batch_input'])
            word_embeddings = tf.cast(word_embeddings, tf.float32)
            if opts.char:
                _, char_embeddings = char_layer.forward(placeholders['char_inp'], placeholders['char_seqlen'], "CharLSTM1")
                char_embeddings = tf.expand_dims(char_embeddings, 0)
                self.embeddings = tf.concat([char_embeddings, word_embeddings], 2)
            else:
                self.embeddings = word_embeddings
            self.hidden_seq_state, self.hidden_last_state = self.blstm_layer.forward(self.embeddings, placeholders['sequence_length'])
            self.unary_potentials = self.ff_layer.forward(self.hidden_seq_state)
            #unary_potentials = tf.reshape(unary_potentials, [config.batch_size, -1, self.num_labels])
            if opts.crf:
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.unary_potentials, placeholders['labels'], placeholders['sequence_length'])
                self.cost =  tf.reduce_mean(-log_likelihood)
            else:
                self.cost = loss(self.unary_potentials, placeholders['labels'])
            self.train_op = train(self.cost)

        #Initialize all variables first as the restore graph doesn't contain optimizer based variables.
        init = tf.variables_initializer([var for var in tf.global_variables() if var.name.startswith(scope)])
        sess.run(init)

        with tf.variable_scope(scope, reuse=True):
            #Restore graph after initialization
            if opts.restore:
                super(POSTagger, self)._initialize(sess, opts)

    def train(self, seqlen, inp, train_from_scratch=False):
        batch_len = len(inp) // config.batch_size
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
                            tag = [0] * self.num_labels
                            tag[label] = 1
                            y[-1].append(tag)
                sess.run(self.train_op, feed_dict={self.placeholders['batch_input']: x,
                                                   self.placeholders['labels']: y, self.placeholders['sequence_length']: seq_len})
                loss_[-1].append(sess.run(self.cost, feed_dict={self.placeholders['batch_input']: x,
                                                                self.placeholders['labels']: y, self.placeholders['sequence_length']: seq_len}))
                print loss_[-1][-1]
        if train_from_scratch:
            loader.save_smodel(sess)


    def eval(self, seqlen, inp):
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
                        tag = [0] * self.num_labels
                        tag[label] = 1
                        y[-1].append(tag)
            if opts.crf:
                trans_mat = sess.run(self.transition_params)
                unary_pot = sess.run(self.unary_potentials, feed_dict={self.placeholders['batch_input']: x,
                                                                       self.placeholders['labels']: y, self.placeholders['sequence_length']: seq_len})
                # CRF decodes only one sequence at a time
                # TODO: Decoding only 1st sequence as batch_size is 1. Change if batch_size increases.
                pred, _ = tf.contrib.crf.viterbi_decode(unary_pot[0], trans_mat)
            else:
                pred = sess.run(self.unary_potentials, feed_dict={self.placeholders['batch_input']: x,
                                                                  self.placeholders['labels']: y, self.placeholders['sequence_length']: seq_len})
            # Only the first sequence since batch_size=1
            if opts.crf:
                for t, p in zip(y[0], pred):
                    print "Predicted ", p
                    print "True ", t
                    predictions.append(p)
                    true_labels.append(t)
            else:
                # TODO: Change design for batch_size>1.
                for t, p in zip(y[0], pred[0]):
                    print "Predicted ", np.argmax(p)
                    print "True ", np.argmax(t)
                    predictions.append(np.argmax(p))
                    true_labels.append(np.argmax(t))

        eval(predictions, true_labels, tag_to_id)


if __name__ == "__main__":

    sess = tf.Session()

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
        "-e", "--char", default=False,
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

    with tf.variable_scope("word"):
        emb_layer = Embedding(opts, word2vec_emb_path, glove_emb_path)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
    seqlen, inp = utils.create_batches(input_x, seqlen, input_y)

    placeholders = {}
    if opts.char:
        # dimension of batch and sequence_len are collapsed as batch_size is 1.
        placeholders['char_inp'] = tf.placeholder("float32", shape=[None, None, len(char_to_id)], name="char_input")
        placeholders['char_seqlen'] = tf.placeholder("int32", shape=[None], name="char_seqlen")

    placeholders['batch_input'] = tf.placeholder("int32", shape=[None, None], name="input")
    placeholders['sequence_length'] = tf.placeholder("int32", shape=[None], name="seqlen")
    if opts.crf:
        placeholders['labels'] = tf.placeholder("int32", shape=[None, None], name="labels")
    else:
        placeholders['labels'] = tf.placeholder("int32", shape=[None, None, num_labels], name="labels")

    graph = loader.reload_smodel(sess, "./source_blstm_crf/", "source_model_crf.meta")
    initialize_embeddings(graph)
    sourcePOS = POSTagger(graph, sess, opts, placeholders, emb_layer, "SourcePOS")
    targetPOS = POSTagger(graph, sess, opts, placeholders, emb_layer, "TargetPOS")

    ##Run model on test data
    input_x, input_y = loader.prepare_input(config.datadir + config.test)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y = utils.convert_tag_to_id(tag_to_id, input_y)
    seqlen, inp = utils.create_batches(input_x, seqlen, input_y)
    sourcePOS.eval(seqlen, inp)

    ##Run on Medpost (target) data
    input_x, input_y = loader.prepare_medpost_input()
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y = utils.convert_tag_to_id(tag_to_id, input_y)
    seqlen, inp = utils.create_batches(input_x, seqlen, input_y)
    targetPOS.eval(seqlen, inp)
