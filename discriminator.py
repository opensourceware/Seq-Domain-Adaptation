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


class AdversarialLearning():

    def __init__(self, sess, graph):
        self.sess = sess
        self.graph = graph
        self.emb_layer = pretrain.Embedding(config.ext_emb_path)
        self.source_lstm = generator.SourceLSTM(graph)
        self.target_lstm = generator.SourceLSTM(graph)

        self.batch_input = graph.get_tensor_by_name("input")
        self.sequence_length = graph.get_tensor_by_name("seqlen")
        self.labels = graph.get_tensor_by_name("labels")

        embeddings = self.emb_layer.lookup(self.batch_input)
        source_hidden_output = self.source_lstm.forward(embeddings, self.sequence_length)
        target_hidden_output = self.target_lstm.forward(embeddings, self.sequence_length)
        self.source_lstm._initialize(sess)
        self.target_lstm._initialize(sess)

        self.discriminator = Discriminator(config.lstm_size*2)
        self.optimizer = tf.AdamOptimizer(lr)
        self.d_loss = None
        self.d_cost = None
        self.g_loss = None
        self.g_cost = None
        d_tvars = [param for param in tf.trainable_variables() if 'discriminator' in param]
        g_tvars = [param for param in tf.trainable_variables() if "TargetLSTM" in param]
        self.discrim_train_op = self.optimizer.minimize(self.d_cost, var_list=d_tvars)
        self.tlstm_train_op = self.optimizer.minimize(self.g_cost, var_list=g_tvars)

    def discrim_loss(self, logits, true_label):
        self.d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_label)
        self.d_cost = tf.reduce_mean(self.d_loss)

    def tlstm_loss(self, predictions):
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
            embeddings = self.emb_layer.lookup(inp)
            if label==0:
                _, hidden_state = self.source_lstm.forward(embeddings, inp_len)
            else:
                _, hidden_state = self.target_lstm.forward(embeddings, inp_len)
            logits = self.discriminator.classify(hidden_state)
            self.discriminator.loss(logits, true_label)
            sess.run(self.discrim_train_op, feed_dict={self.batch_input:inp, self.sequence_length:inp_len})

    def tlstm_train(self, input_x, seqlen):
        for i in range(5):
            ind, inp = utils.get_batch(input_x)
            inp_len = seqlen[ind]
            embeddings = self.emb_layer.lookup(inp)
            _, hidden_state = self.target_lstm.forward(embeddings, inp_len)
            logits = self.discriminator.classify(hidden_state)
            self.tlstm_loss(logits)
            sess.run(self.tlstm_train_op, feed_dict={self.batch_input:inp, self.sequence_length:inp_len})

    def train(self):
        input_x, input_y = loader.prepare_input(config.datadir + config.train)
        seqlen, input_x = utils.convert_to_id(input_x, self.emb_layer.word_to_id)
        input_y, tag_to_id = utils.convert_tag_to_id(input_y)
        seqlen, inp = utils.create_batches(input_x, input_y, seqlen, config.batch_size)
        input_x = [[seq[0] for seq in batch] for batch in inp]
        s_len = len(input_x)
        t_len = len()
        for _ in range(config.num_epochs):
            for i in range(t_len):
                self.tlstm_train(input_x, seqlen)
                self.discrim_train(input_x, seqlen)


if __name__ == "__main__":
    sess = tf.Session()
    graph = loader.reload_smodel(sess)
    adv = AdversarialLearning(sess, graph)

