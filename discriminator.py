import tensorflow as tf
import numpy as np
import config, pretrain, generator, loader, utils
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
            xavier_init_1 = 1.0 / np.sqrt(config.conv_filter_height*config.conv1_filter_width)
            xavier_init_2 = 1.0 / np.sqrt(config.conv_filter_height*config.conv2_filter_width)
            xavier_init_3 = 1.0 / np.sqrt(config.conv_filter_height*config.conv3_filter_width)
            self.filter1 = tf.Variable(tf.truncated_normal([config.conv_filter_height, config.conv1_filter_width, 1, 1], mean=0.0,
                                                       stddev=xavier_init_1), name='convlayer1')
            self.filter2 = tf.Variable(tf.truncated_normal([config.conv_filter_height, config.conv2_filter_width, 1, 1], mean=0.0,
                                                       stddev=xavier_init_2), name='convlayer2')
            self.filter3 = tf.Variable(tf.truncated_normal([config.conv_filter_height, config.conv3_filter_width, 1, 1], mean=0.0,
                                                       stddev=xavier_init_3), name='convlayer3')
            self.classifier_weights = tf.Variable(tf.random_normal([60, 2], stddev=0.035, dtype=tf.float32), name="discrim_weights", trainable=True)
            self.classifier_bias = tf.Variable(tf.zeros(2, dtype=tf.float32), name="discrim_bias", trainable=True)

    def forward(self, seq_hidden_state, seq_len):
        seq_hidden_state = tf.expand_dims(seq_hidden_state, -1)
        #if seq_len[0]<4:
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
        self.prediction = tf.add(tf.matmul(tf.squeeze(self.conv_output, [-1]), self.classifier_weights), self.classifier_bias)
        return self.prediction[0]


class AdversarialLearning(object):

    def __init__(self, sess, opts):
        self.sess = sess

        self.batch_input = tf.placeholder("int32", shape=[None, None], name="input")
        self.sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
        self.label = tf.placeholder(tf.bool, shape=[2], name="labels")

        self.emb_layer = pretrain.Embedding(opts, config.word2vec_emb_path, config.glove_emb_path)
        self.source_lstm = generator.SourceLSTM()
        embeddings = self.emb_layer.lookup(self.batch_input)
        embeddings = tf.cast(embeddings, tf.float32)
        self.source_seq_state, self.source_last_state = self.source_lstm.forward(embeddings, self.sequence_length)
        #Restore source LSTM after SourceLSTM variables are created i.e. the weights are 
        #automatically loaded in SourceLSTM variables from the checkpoint.
        saver = tf.train.Saver()
        saver.restore(sess, "./source_model_only_embeddings/source_model_only_embeddings")
        #Now create the target LSTM and initialize from the weights in the saved checkpoint.
        self.target_lstm = generator.TargetLSTM()
        self.target_seq_state, self.target_last_state = self.target_lstm.forward(embeddings, self.sequence_length)

        self.target_lstm._initialize(sess)

        self.discriminator = Discriminator()
        self.discrim_logits = tf.cond(self.label[1], lambda: self.discriminator.forward(self.target_seq_state, self.sequence_length),
                    lambda: self.discriminator.forward(self.source_seq_state, self.sequence_length))
        self.tlstm_logits = self.discriminator.forward(self.target_seq_state, self.sequence_length)

        #Can fix the learning rate in AdamOptimizer because the final gradient updates decay in the formula.
        self.optimizer = tf.train.AdamOptimizer(0.005)
        self.discrim_loss(self.discrim_logits, self.label)
        self.tlstm_loss(self.tlstm_logits)
        self.d_tvars = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        self.g_tvars = [param for param in tf.trainable_variables() if "TargetLSTM" in param.name]
        self.discrim_train_op = self.optimizer.minimize(self.d_cost, var_list=self.d_tvars)
        self.tlstm_train_op = self.optimizer.minimize(self.g_cost, var_list=self.g_tvars)


    def discrim_loss(self, logits, true_label):
        self.d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_label)
        self.d_cost = tf.reduce_mean(self.d_loss)


    def tlstm_loss(self, predictions):
        #Target LSTM tries to maximally confuse the discriminator.
        self.g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=[0.5, 0.5])
        self.g_cost = tf.reduce_mean(self.g_loss)


    def discrim_train(self, s_input, t_input, s_seqlen, t_seqlen):
        for i in range(1):
            label = np.random.randint(2)
            true_label = [0, 0]
            true_label[label] = 1
            ##TODO: Fix for batch_size>1
            #true_label = true_label*config.batch_size
            true_label = [bool(a) for a in true_label]
            if label==0:
                ind, inp = utils.get_batch(s_input)
                inp_len = s_seqlen[ind]
            else:
                ind, inp = utils.get_batch(t_input)
                inp_len = t_seqlen[ind]
            self.sess.run(self.discrim_train_op, feed_dict={self.batch_input:inp, self.sequence_length:inp_len, self.label:true_label})
        return self.sess.run(self.d_cost, feed_dict={self.batch_input:inp, self.sequence_length:inp_len, self.label:true_label})


    def tlstm_train(self, input_x, seqlen, num_updates=5):
        for i in range(num_updates):
            ind, inp = utils.get_batch(input_x)
            inp_len = seqlen[ind]
            self.sess.run(self.tlstm_train_op, feed_dict={self.batch_input:inp, self.sequence_length:inp_len})
        return self.sess.run(self.g_cost, feed_dict={self.batch_input:inp, self.sequence_length:inp_len})


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

	sess = tf.Session()

	adv = discriminator.AdversarialLearning(sess, opts)

	input_x, _ = loader.prepare_input(config.datadir + config.train)
	s_seqlen, s_input = utils.convert_to_id(input_x, adv.emb_layer.word_to_id)
	s_seqlen, s_input = utils.create_batches(s_input, s_seqlen)
	input_x, _ = loader.prepare_medpost_input()
	t_seqlen, t_input = utils.convert_to_id(input_x, adv.emb_layer.word_to_id)
	t_seqlen, t_input = utils.create_batches(t_input, t_seqlen)
	s_len = len(s_input)
	t_len = len(t_input)

	#Do not initialize Source and Target LSTM weights; The variables are from index 0 to 7.
	#TODO: Find better fix for initialization of variables
	init = tf.variables_initializer(tf.global_variables()[9:])
	sess.run(init)

	gloss = []
	dloss = []
	plt.axis([0, 10000, 0, 4])
	plt.ion()
	train_steps = 0
	for epoch in range(config.num_epochs):
	    for i in range(t_len/(10*config.batch_size)):
		print epoch
		gloss.append(adv.tlstm_train(t_input, t_seqlen))
		dloss.append(adv.discrim_train(s_input, t_input, s_seqlen, t_seqlen))
		train_steps += 1
		if gloss[-1]>1.5:
			print "Train only generator with 300 iterations (1500 updates)"
			gloss.append(adv.tlstm_train(t_input, t_seqlen, 300))
		print train_steps
		print gloss[-1]
		print dloss[-1]

	for g,d,n in zip(gloss, dloss, range(7700)):
	    plt.scatter(n, g, color="r")
	    plt.scatter(n, d, color="b")

	saver = tf.train.Saver([tf.global_variables()[i] for i in range(5,9)])
	saver.save(sess, "./target_model")

