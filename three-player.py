import tensorflow as tf
import pretrain, utils, loader, config
import optparse

sess = tf.Session()

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
input_x, input_y = loader.prepare_input(config.datadir + config.train)
if opts.char:
    char_emb, char_to_id, char_seq_len = utils.convert_to_char_emb(input_x)

with tf.variable_scope("word"):
    emb_layer = pretrain.Embedding(opts, word2vec_emb_path, glove_emb_path)

opts.restore = False

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
    placeholders['labels'] = tf.placeholder("int32", shape=[None, None, len(tag_to_id)], name="labels")

targetPOS = pretrain.POSTagger(sess, opts, placeholders, emb_layer, "TargetPOS")
saver = tf.train.Saver()
saver.restore(sess, "./3-player-t_model/target_model")
##Run on Medpost (target) data
input_x, input_y = loader.prepare_medpost_input()
seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
input_y = utils.convert_tag_to_id(tag_to_id, input_y)
seqlen, inp = utils.create_batches(input_x, seqlen, input_y)
targetPOS.eval(seqlen, inp, sess, opts)


sourcePOS = pretrain.POSTagger(sess, opts, placeholders, emb_layer, "SourcePOS")
saver = tf.train.Saver()
saver.restore(sess, "./3-player-s_model/source_model")
##Run model on test data
input_x, input_y = loader.prepare_input(config.datadir + config.test)
seqlen, input_x = utils.convert_to_id(input_x, emb_layer.word_to_id)
input_y = utils.convert_tag_to_id(tag_to_id, input_y)
seqlen, inp = utils.create_batches(input_x, seqlen, input_y)
sourcePOS.eval(seqlen, inp, sess, opts)
