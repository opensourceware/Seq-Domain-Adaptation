import tensorflow as tf
import numpy as np
import config, pretrain, generator, loader, utils

class Discriminator:
    """
    A binary classifier which discrminates between two domains.
    """
    def __init__(self, hidden_size):
        with tf.variable_scope('discriminator'):
            self.classifier_weights = tf.Variable(tf.random_normal([hidden_size, 2], stddev=0.035, dtype=tf.float64), name="discrim_weights", trainable=True)
            self.classifier_bias = tf.Variable(tf.zeros(2, dtype=tf.float64), name="discrim_bias", trainable=True)
            self.lr = tf.placeholder("float64", shape=None)
            self.prediction = None
            self.loss = None
            self.cost = None

    def classify(self, hidden_inp):
        self.prediction = tf.add(tf.matmul(hidden_inp, self.classifier_weights), self.classifier_bias)
        return  self.prediction


class AdversarialLearning(object):

    def __init__(self, sess):
        self.sess = sess

        self.batch_input = tf.placeholder("int32", shape=[None, None], name="input")
        self.sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
        self.label = tf.placeholder(tf.bool, shape=[2], name="labels")

        self.emb_layer = pretrain.Embedding(config.ext_emb_path)
        self.source_lstm = generator.SourceLSTM()
        embeddings = self.emb_layer.lookup(self.batch_input)
        _, self.source_last_state = self.source_lstm.forward(embeddings, self.sequence_length)
        #Restore source LSTM after SourceLSTM variables are created.
        saver = tf.train.Saver()
        saver.restore(sess, "./source_model")
        #Now create the target LSTM and initialize from the weights in the saved checkpoint.
        self.target_lstm = generator.TargetLSTM()
        _, self.target_last_state = self.target_lstm.forward(embeddings, self.sequence_length)
        self.target_lstm._initialize(sess)

        self.discriminator = Discriminator(config.lstm_size*2)
        self.discrim_logits = tf.cond(self.label[1], lambda: self.discriminator.classify(self.target_last_state),
                    lambda: self.discriminator.classify(self.source_last_state))
        self.tlstm_logits = self.discriminator.classify(self.target_last_state)

        #Can fix the learning rate in AdamOptimizer because the final gradient updates decay in the formula.
        self.optimizer = tf.train.AdamOptimizer(0.005)
        self.discrim_loss(self.discrim_logits, self.label)
        self.tlstm_loss(self.tlstm_logits)
        d_tvars = [param for param in tf.trainable_variables() if 'discriminator' in param]
        g_tvars = [param for param in tf.trainable_variables() if "TargetLSTM" in param]
        self.discrim_train_op = self.optimizer.minimize(self.d_cost, var_list=d_tvars)
        self.tlstm_train_op = self.optimizer.minimize(self.g_cost, var_list=g_tvars)


    def discrim_loss(self, logits, true_label):
        self.d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_label)
        self.d_cost = tf.reduce_mean(self.d_loss)


    def tlstm_loss(self, predictions):
        #Target LSTM tries to maximally confuse the discriminator.
        self.g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=[0.5, 0.5])
        self.g_cost = tf.reduce_mean(self.g_loss)


    def discrim_train(self, s_input, t_input, s_seqlen, t_seqlen):
        for i in range(5):
            label = np.random.randint(2)
            true_label = [0, 0]
            true_label[label] = 1
            true_label = true_label*config.batch_size
            if label==0:
                ind, inp = utils.get_batch(s_input)
                inp_len = s_seqlen[ind]
            else:
                ind, inp = utils.get_batch(t_input)
                inp_len = t_seqlen[ind]
            sess.run(self.discrim_train_op, feed_dict={self.batch_input:inp, self.sequence_length:inp_len})


    def tlstm_train(self, input_x, seqlen):
        for i in range(5):
            ind, inp = utils.get_batch(input_x)
            inp_len = seqlen[ind]
            sess.run(self.tlstm_train_op, feed_dict={self.batch_input:inp, self.sequence_length:inp_len})

    def train(self):
        input_x, _ = loader.prepare_input(config.datadir + config.train)
        s_seqlen, s_input = utils.convert_to_id(input_x, self.emb_layer.word_to_id)
        input_x, _ = loader.prepare_medpost_input()
        t_seqlen, t_input = utils.convert_to_id(input_x, self.emb_layer.word_to_id)

        s_len = len(s_input)
        t_len = len(t_input)
        for _ in range(config.num_epochs):
            for i in range(t_len/(10*config.batch_size)):
                self.tlstm_train(s_input, s_seqlen)
                self.discrim_train(s_input, t_input, s_seqlen, t_seqlen)


if __name__ == "__main__":
    sess = tf.Session()
    batch_size = config.batch_size
    ext_emb_path = config.ext_emb_path
    input_x, input_y = loader.prepare_input(config.datadir + config.train)
    emb_layer = pretrain.Embedding(ext_emb_path)
    seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
    input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
    seqlen, inp = utils.create_batches(input_x, input_y, seqlen, batch_size)

    num_labels = len(tag_to_id)

    graph = loader.reload_smodel(sess)
    adv = AdversarialLearning(sess)

