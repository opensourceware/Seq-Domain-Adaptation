import tensorflow as tf
import config, loader, utils, pretrain


class BaseLSTM(object):
    """
    Base class for bi-directional LSTM classes which are initialized to the the same weights.
    """
    def __init__(self, scope):
        """
        Load all BLSTM weights and biases from the pre-computed graph.
        :param graph: Graph loaded from saved checkpoint.
        """
        with tf.variable_scope(scope, reuse=True):
            self.pretrain_lstm_fw_weights = tf.get_variable("bidirectional_rnn/fw/lstm_cell/weights", dtype="float32")
            self.pretrain_lstm_fw_biases = tf.get_variable("bidirectional_rnn/fw/lstm_cell/biases", dtype="float32")
            self.pretrain_lstm_bw_weights = tf.get_variable("bidirectional_rnn/bw/lstm_cell/weights", dtype="float32")
            self.pretrain_lstm_bw_biases = tf.get_variable("bidirectional_rnn/bw/lstm_cell/biases", dtype="float32")

    def _initialize(self, sess):
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


class SourceLSTM:
    """
    Source LSTM contains LSTM cell trained on PTB data. This LSTM cell is not trained during Adversarial training.
    """
    def __init__(self):
        ##No need for max num of time steps
        self.lstm_size = config.lstm_size
        self.cell_fw = tf.contrib.rnn.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)
        self.cell_bw = tf.contrib.rnn.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

    def forward(self, input, input_length):
        if config.keep_prob < 1:
            input = tf.nn.dropout(input, config.keep_prob)
        with tf.variable_scope("SourceLSTM"):
            output, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                                             dtype=tf.float32, sequence_length=input_length,
                                                             inputs=input)
            output = tf.concat(output, 2)
            last_state = tf.concat(last_state, 2)
        #last_state contains both cell state and hidden state. Return only hidden_state.
        return output, last_state[1]


class TargetLSTM(BaseLSTM):
    """
    Target LSTM contains LSTM cell which will be trained to adapt to sequence (POS) tagging task for target domain (medical data).
    The weights are learned during adversarial training in which the TargetLSTM tries to mimic SourceLSTM's output distribution.
    """
    def __init__(self):
        ##No need for max num of time steps
        super(TargetLSTM, self).__init__("SourceLSTM")
        self.lstm_size = config.lstm_size
        self.cell_fw = tf.contrib.rnn.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)
        self.cell_bw = tf.contrib.rnn.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

    def forward(self, input, input_length):
        if config.keep_prob < 1:
            input = tf.nn.dropout(input, config.keep_prob)
        with tf.variable_scope("TargetLSTM"):
            output, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                                             dtype=tf.float32, sequence_length=input_length,
                                                             inputs=input)
            output = tf.concat(output, 2)
            last_state = tf.concat(last_state, 2)
        #last_state contains both cell state and hidden state. Return only hidden_state.
        return output, last_state[1]

    def _initialize(self, sess):
        with tf.variable_scope("TargetLSTM", reuse=True):
            super(TargetLSTM, self)._initialize(sess)


def main():
    batch_size = 10
    ext_emb_path = config.ext_emb_path
    input_x, input_y = loader.prepare_input(config.datadir+config.train)
    emb_layer = pretrain.Embedding(ext_emb_path)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y, tag_to_id = utils.convert_tag_to_id(input_y)
    seqlen, inp = utils.create_batches(input_x, input_y, seqlen, batch_size)

    sess = tf.Session()
    graph = loader.reload_smodel(sess)
    num_labels = len(tag_to_id)
    source_lstm = SourceLSTM()
    target_lstm = TargetLSTM()
    ff_layer = pretrain.FeedForward(2*config.lstm_size, num_labels)

    init_op = tf.global_variables_initializer()
    batch_input = tf.placeholder("int32", shape=[None, None], name="input")
    sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
    labels = tf.placeholder("int32", shape=[None, None, num_labels], name="labels")

    embeddings = emb_layer.lookup(batch_input)
    source_hidden_output = source_lstm.forward(embeddings, sequence_length)
    target_hidden_output = target_lstm.forward(embeddings, sequence_length)

    #sess.run(init_op)
    target_lstm._initialize(sess)


if __name__ == "__main__":
    main()
