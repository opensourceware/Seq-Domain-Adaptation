import tensorflow as tf
import numpy as np
import config, pretrain, lstm_mapper, loader, utils
import matplotlib.pyplot as plt

"""
class Discriminator:
    
    A binary classifier which discrminates between two domains.
    
    def __init__(self, hidden_size):
        with tf.variable_scope('discriminator'):
            self.classifier_weights = tf.Variable(tf.random_normal([hidden_size, 2], stddev=0.035, dtype=tf.float32), name="discrim_weights", trainable=True)
            self.classifier_bias = tf.Variable(tf.zeros(2, dtype=tf.float32), name="discrim_bias", trainable=True)
            self.lr = tf.placeholder("float32", shape=None)
            self.prediction = None
            self.loss = None
            self.cost = None

    def classify(self, hidden_inp):
        self.prediction = tf.add(tf.matmul(hidden_inp, self.classifier_weights), self.classifier_bias)
        return  self.prediction
"""


class Discriminator:
    def __init__(self, ):
        with tf.variable_scope('discriminator'):
            xavier_init_1 = 1.0 / np.sqrt(config.conv_filter_height * config.conv1_filter_width)
            xavier_init_2 = 1.0 / np.sqrt(config.conv_filter_height * config.conv2_filter_width)
            xavier_init_3 = 1.0 / np.sqrt(config.conv_filter_height * config.conv3_filter_width)
            self.filter1 = tf.Variable(
                tf.truncated_normal([config.conv_filter_height, config.conv1_filter_width, 1, 1], mean=0.0,
                                    stddev=xavier_init_1), name='convlayer1')
            self.filter2 = tf.Variable(
                tf.truncated_normal([config.conv_filter_height, config.conv2_filter_width, 1, 1], mean=0.0,
                                    stddev=xavier_init_2), name='convlayer2')
            self.filter3 = tf.Variable(
                tf.truncated_normal([config.conv_filter_height, config.conv3_filter_width, 1, 1], mean=0.0,
                                    stddev=xavier_init_3), name='convlayer3')
            self.classifier_weights = tf.Variable(tf.random_normal([60, 2], stddev=0.035, dtype=tf.float32),
                                                  name="discrim_weights", trainable=True)
            self.classifier_bias = tf.Variable(tf.zeros(2, dtype=tf.float32), name="discrim_bias", trainable=True)

    def forward(self, seq_hidden_state):
        seq_hidden_state = tf.expand_dims(seq_hidden_state, -1)
        # if seq_len[0]<4:
        #    seq_hidden_state = tf.concat([seq_hidden_state, tf.zeros([1, 200, (4-seq_len), 1])], axis=2)
        self.conv1 = tf.nn.conv2d(seq_hidden_state, self.filter1, padding='SAME', strides=[1, 1, 10, 1])
        self.conv2 = tf.nn.conv2d(seq_hidden_state, self.filter2, padding='SAME', strides=[1, 1, 10, 1])
        self.conv3 = tf.nn.conv2d(seq_hidden_state, self.filter3, padding='SAME', strides=[1, 1, 10, 1])
        self.conv1 = tf.nn.relu(self.conv1)
        self.conv2 = tf.nn.relu(self.conv2)
        self.conv3 = tf.nn.relu(self.conv3)
        self.maxpool1 = tf.reduce_max(self.conv1, axis=1)
        self.maxpool2 = tf.reduce_max(self.conv2, axis=1)
        self.maxpool3 = tf.reduce_max(self.conv3, axis=1)
        self.conv_output = tf.concat([self.maxpool1, self.maxpool2, self.maxpool3], axis=1)
        self.prediction = tf.add(tf.matmul(tf.squeeze(self.conv_output, [-1]), self.classifier_weights),
                                 self.classifier_bias)
        return self.prediction[0]


class AdversarialLearning(object):
    def __init__(self, sess, opts, num_labels):
        self.sess = sess

        self.batch_input = tf.placeholder("int32", shape=[None, None], name="input")
        self.sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
        if opts.crf:
            self.labels = tf.placeholder("int32", shape=[None, None], name="labels")
        else:
            self.labels = tf.placeholder("int32", shape=[None, None, num_labels], name="labels")

        self.discrim_label = tf.placeholder(tf.bool, shape=[2], name="labels")

        placeholders = {'batch_input': self.batch_input,
                        'sequence_length': self.sequence_length,
                        'labels': self.labels,
                        'discrim_labels': self.discrim_label}

        self.emb_layer = pretrain.Embedding(opts, config.word2vec_emb_path, config.glove_emb_path)
        self.source_pos = pretrain.POSTagger(sess, opts, placeholders, self.emb_layer, "SourcePOS")
        self.target_pos = pretrain.POSTagger(sess, opts, placeholders, self.emb_layer, "TargetPOS")

        self.discriminator = Discriminator()
        self.discrim_logits = tf.cond(self.discrim_label[1],
                                      lambda: self.discriminator.forward(self.target_pos.hidden_seq_state),
                                      lambda: self.discriminator.forward(self.source_pos.hidden_seq_state))
        self.tlstm_logits = self.discriminator.forward(self.target_pos.hidden_seq_state)
        self.slstm_logits = self.discriminator.forward(self.source_pos.hidden_seq_state)

        # Can fix the learning rate in AdamOptimizer because the final gradient updates decay in the formula.
        self.optimizer = tf.train.AdamOptimizer(0.005)
        self.discrim_loss(self.discrim_logits, self.discrim_label)
        self.tlstm_loss(self.tlstm_logits)
        self.slstm_loss(self.slstm_logits)
        self.d_tvars = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        self.t_tvars = [param for param in tf.trainable_variables() if "TargetPOS/bidirectional_rnn" in param.name]
        self.s_vars = [param for param in tf.trainable_variables() if "SourcePOS/bidirectional_rnn" in param.name]
        self.discrim_train_op = self.optimizer.minimize(self.d_cost, var_list=self.d_tvars)
        self.tlstm_train_op = self.optimizer.minimize(self.t_cost, var_list=self.t_tvars)
        self.slstm_train_op = self.optimizer.minimize(self.s_cost, var_list=self.s_vars)

    def discrim_loss(self, logits, true_label):
        self.d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_label)
        self.d_cost = tf.reduce_mean(self.d_loss)

    def tlstm_loss(self, predictions):
        # Target LSTM tries to maximally confuse the discriminator.
        self.t_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=[0.5, 0.5])
        self.t_cost = tf.reduce_mean(self.t_loss)

    def slstm_loss(self, predictions):
        # Source LSTM also tries to maximally confuse the discriminator to assist the TargetLSTN.
        self.s_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=[0.5, 0.5])
        self.s_cost = tf.reduce_mean(self.s_loss)

    def discrim_train(self, s_input, t_input, s_seqlen, t_seqlen):
        for i in range(1):
            label = np.random.randint(2)
            true_label = [0, 0]
            true_label[label] = 1
            ##TODO: Fix for batch_size>1
            # true_label = true_label*config.batch_size
            true_label = [bool(a) for a in true_label]
            if label == 0:
                ind, inp = utils.get_batch(s_input)
                inp_len = s_seqlen[ind]
            else:
                ind, inp = utils.get_batch(t_input)
                inp_len = t_seqlen[ind]
            self.sess.run(self.discrim_train_op,
                          feed_dict={self.batch_input: inp[0][0], self.sequence_length: inp_len, self.discrim_label: true_label})
        return self.sess.run(self.d_cost,
                             feed_dict={self.batch_input: inp[0][0], self.sequence_length: inp_len, self.discrim_label: true_label})

    def tlstm_train(self, inp, seqlen, num_updates=5):
        for i in range(num_updates):
            ind, inp = utils.get_batch(inp)
            inp_len = seqlen[ind]
            self.sess.run(self.tlstm_train_op, feed_dict={self.batch_input: inp[0][0], self.sequence_length: inp_len})
            self.sess.run(self.target_pos.train_op,
                          feed_dict={self.batch_input: inp[0][0], self.sequence_length: inp_len,
                                     self.labels: inp[0][1]})
        return self.sess.run(self.t_cost, feed_dict={self.batch_input: inp, self.sequence_length: inp_len})

    def slstm_train(self, inp, seqlen, num_updates=5):
        for i in range(num_updates):
            ind, inp = utils.get_batch(inp)
            inp_len = seqlen[ind]
            self.sess.run(self.slstm_train_op, feed_dict={self.batch_input: inp[0][0], self.sequence_length: inp_len})
            self.sess.run(self.source_pos.train_op,
                          feed_dict={self.batch_input: inp[0][0], self.sequence_length: inp_len,
                                     self.labels: inp[0][1]})
        return self.sess.run(self.s_cost, feed_dict={self.batch_input: inp, self.sequence_length: inp_len})

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
        "-r", "--restore", default=True,
        help="Rebuild the model and restore weights from checkpoint"
    )
    opts = optparser.parse_args()[0]

    input_x, input_y = loader.prepare_input(config.datadir + config.train)
    seqlen, input_x = utils.convert_to_id(input_x, adv.emb_layer.word_to_id)
    input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
    s_seqlen, s_inp = utils.create_batches(input_x, seqlen, input_y)

    input_x, input_y = loader.prepare_medpost_input()
    seqlen, input_x = utils.convert_to_id(input_x, adv.emb_layer.word_to_id)
    input_y = utils.convert_tag_to_id(tag_to_id, input_y)
    t_seqlen, t_inp = utils.create_batches(input_x, seqlen, input_y)

    s_len = len(s_inp)
    t_len = len(t_inp)

    sess = tf.Session()

    adv = AdversarialLearning(sess, opts, len(tag_to_id))

    # Do not initialize Source and Target POS tagger weights
    init_vars = [v for v in tf.global_variables() if
                 not v.name.startswith("SourcePOS") and not v.name.startswith("TargetPOS")]
    init = tf.variables_initializer(init_vars)
    sess.run(init)

    tloss = []
    dloss = []
    sloss = []
    plt.axis([0, 10000, 0, 4])
    plt.ion()
    train_steps = 0
    for epoch in range(config.num_epochs):
        for i in range(t_len / (10 * config.batch_size)):
            print epoch
            tloss.append(adv.tlstm_train(t_inp, t_seqlen))
            dloss.append(adv.discrim_train(s_inp, t_inp, s_seqlen, t_seqlen))
            sloss.append(adv.slstm_train(s_inp, s_seqlen))
            train_steps += 1
            #if tloss[-1] > 1.5:
            #    print "Train only mapper with 300 iterations (1500 updates)"
            #    tloss.append(adv.tlstm_train(t_inp, t_seqlen, 300))
            print train_steps
            print tloss[-1]
            print dloss[-1]
            print sloss[-1]

    for g, d, n in zip(tloss, dloss, range(7700)):
        plt.scatter(n, g, color="r")
        plt.scatter(n, d, color="b")

    saver = tf.train.Saver([tf.global_variables()[i] for i in range(5, 9)])
    saver.save(sess, "./target_model")
