
import tensorflow as tf
import numpy as np
import config, pretrain, lstm_mapper, loader, utils
import matplotlib.pyplot as plt
import optparse

optparser = optparse.OptionParser()
optparser.add_option(
	"-g", "--glove", default=True,
	help="Use glove embeddings"
)
optparser.add_option(
	"-c", "--crf", default=False,
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


tf.reset_default_graph()
sess = tf.Session()
batch_size = config.batch_size
word2vec_emb_path = config.word2vec_emb_path
glove_emb_path = config.glove_emb_path
input_x, input_y = loader.prepare_input(config.datadir+config.train)
emb_layer = pretrain.Embedding(opts, word2vec_emb_path, glove_emb_path)
seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
input_y, tag_to_id = utils.create_and_convert_tag_to_id(input_y)
seqlen, inp = utils.create_batches(input_x, seqlen, input_y)

num_labels = len(tag_to_id)
lstm_size = 100
blstm_layer = pretrain.BLSTM(lstm_size)

batch_input = tf.placeholder("int32", shape=[None, None], name="input")
sequence_length = tf.placeholder("int32", shape=[None], name="seqlen")
if opts.crf:
	labels = tf.placeholder("int32", shape=[None, None],  name="labels")
else:
	labels = tf.placeholder("int32", shape=[None, None, num_labels],  name="labels")

embeddings = emb_layer.lookup(batch_input)
embeddings = tf.cast(embeddings, tf.float32)
hidden_output, _ = blstm_layer.forward(embeddings, sequence_length, "SourceLSTM")

ff_layer = pretrain.FeedForward(2*lstm_size, num_labels)
unary_potentials = ff_layer.forward(hidden_output)
if opts.crf:
	log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_potentials, labels, sequence_length)
	cost =  tf.reduce_mean(-log_likelihood)
else:
	cost = pretrain.loss(unary_potentials, labels)

train_op = pretrain.train(cost)

if opts.restore:
    saver = tf.train.Saver()
    saver.restore(sess, "./source_model_only_embeddings/source_model_only_embeddings")
else:
    init = tf.global_variables_initializer()
    sess.run(init)

graph = loader.reload_smodel(sess)

pretrain_lstm_fw_weights = graph.get_tensor_by_name("TargetLSTM/bidirectional_rnn/fw/lstm_cell/weights:0")
pretrain_lstm_fw_biases = graph.get_tensor_by_name("TargetLSTM/bidirectional_rnn/fw/lstm_cell/biases:0")
pretrain_lstm_bw_weights = graph.get_tensor_by_name("TargetLSTM/bidirectional_rnn/bw/lstm_cell/weights:0")
pretrain_lstm_bw_biases = graph.get_tensor_by_name("TargetLSTM/bidirectional_rnn/bw/lstm_cell/biases:0")


with tf.variable_scope("SourceLSTM", reuse=True):
	with tf.variable_scope("bidirectional_rnn"):
	    with tf.variable_scope("fw"):
	        with tf.variable_scope("lstm_cell"):
	            lstm_fw_weights = tf.get_variable("weights", dtype="float32")
	            lstm_fw_biases = tf.get_variable("biases", dtype="float32")
	            sess.run(tf.assign(lstm_fw_weights, pretrain_lstm_fw_weights))
	            sess.run(tf.assign(lstm_fw_biases, pretrain_lstm_fw_biases))
	    with tf.variable_scope("bw"):
	        with tf.variable_scope("lstm_cell"):
	            lstm_bw_weights = tf.get_variable("weights", dtype="float32")
	            lstm_bw_biases = tf.get_variable("biases", dtype="float32")
	            sess.run(tf.assign(lstm_bw_weights, pretrain_lstm_bw_weights))
	            sess.run(tf.assign(lstm_bw_biases, pretrain_lstm_bw_biases))



input_x, input_y = loader.prepare_medpost_input()
seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
input_y = utils.convert_tag_to_id(tag_to_id, input_y)
seqlen, inp = utils.create_batches(input_x, seqlen, input_y)

predictions = []
true_labels = []
for seq_len, batch in zip(seqlen[40:], inp[40:]):
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
        #CRF ddecodes only one sequence at a time. 
        pred, _ = tf.contrib.crf.viterbi_decode(unary_pot[0], trans_mat)
    else:
        pred = sess.run(unary_potentials, feed_dict={batch_input: x, labels: y, sequence_length: seq_len})
    #y[0] and pred[0] because batch_size=1
    if opts.crf:
        for t, p in zip(y[0], pred):
            #print "Predicted ", p
            #print "True ", t
            predictions.append(p)
            true_labels.append(t)
    else:
        pred.shape = (len(y), pred.shape[1] / len(y), 45)
        for y_, pred_ in zip(y, pred):
            for t, p in zip(y_, pred_):
                #print "Predicted ", np.argmax(p)
                #print "True ", np.argmax(t)
                predictions.append(np.argmax(p))
                true_labels.append(np.argmax(t))


print utils.eval(predictions, true_labels, tag_to_id)
