# import nltk, os
# from nltk.parse import stanford
# import gensim
import config
import numpy as np
import tensorflow as tf


def create_input(sentence, word_to_id):
    tokens = [token for token in sentence.split()]
    input = [word_to_id[token] if token in word_to_id else 'UNK' for token in tokens]
    return input


def word_to_index(vocab, model=None):
    """
    Indexes vocabulary words and creates weight vector of corresponding words in the embedding file.
    Arg:
        vocab - list of words in the training set
        model - gensim Word2Vec model
    Returns:
        weights - list of embeddings of corresponding words in index
        index - dictionary of indexed vocab
    """
    if model is None:
        pretrained_loc = config.PRETRAINED_WORDS
        with open(pretrained_loc, 'r') as f:
            pretrained_words = f.read().split('\n')
    weights = []
    index = {}
    index['PAD'] = 0
    i = 1
    for word in vocab:
        if word in model:
            weights.append(model[word])
            index[word] = i
            i += 1
    index['UNK'] = i
    emb_size = model['the'].shape[0]
    unk = np.random.normal(0.001, 0.01, emb_size)
    pad = np.zeros(shape=emb_size)
    weights = np.vstack((pad, weights, unk))
    return weights, index


def convert_to_id(input_x, word_to_id):
    seqlen = []
    maxseqlen = 0
    for item in input_x:
        if len(item) > maxseqlen:
            maxseqlen = len(item)
    Idx = []
    for sent in input_x:
        sentIdx = []
        for word in sent:
            try:
                sentIdx.append(word_to_id[word])
            except KeyError:
                sentIdx.append(word_to_id['UNK'])
        seqlen.append(len(sent))
        if seqlen[-1] < maxseqlen:
            sentIdx += [word_to_id['PAD']] * (maxseqlen - seqlen[-1])
        Idx.append(sentIdx)
    return maxseqlen, seqlen, Idx


def convert_tag_to_id(tags, maxseqlen):
    tag_to_id = {}
    tag_to_id['PAD'] = 0
    i = 1
    Idx = []
    for sent in tags:
        sentIdx = []
        for tag in sent:
            if tag not in tag_to_id:
                tag_to_id[tag] = i
                i += 1
            sentIdx.append(tag_to_id[tag])
        if len(sent) < maxseqlen:
            sentIdx += [tag_to_id['PAD']] * (maxseqlen - len(sent))
        Idx.append(sentIdx)
    return Idx, tag_to_id


def create_batches(input_x, input_y, seqlen, batch_size, maxseqlen):
    """
    :param input_x:
    :param input_y:
    :param seqlen:
    :param batch_size:
    :param maxseqlen:
    :return:
    """
    inp = []
    batchseqlen = []
    for num, item in enumerate(input_x):
        inp.append([item, input_y[num]])
    #dico = dict(zip(range(len(inp)), seqlen))
    #inp = sorted(inp, key=lambda x:len(x[0]))
    #seqlen = sorted(seqlen)
    batch_len = len(input_x) // batch_size
    #inp = tf.convert_to_tensor(inp, name="input_data_x", dtype=tf.int32)
    inp, seqlen = pad_last_batch(inp, seqlen, maxseqlen, len(inp), batch_size)
    inp = np.array(inp)
    inp = np.reshape(inp, [batch_len + 1, batch_size, 2, maxseqlen])
    print len(seqlen)
    seqlen = np.reshape(seqlen, [batch_len+1, batch_size])
    return inp, seqlen


def pad_last_batch(inp, seqlen, maxseqlen, inp_len, batch_size):
    num_unbatched = inp_len % batch_size
    inp += [[[0]*maxseqlen, [0]*maxseqlen]]*(batch_size-num_unbatched)
    seqlen += [0]*(batch_size-num_unbatched)
    return inp, seqlen

#def pad_last_batch(inp, batch_len, batch_size):
#    num_unbatched = batch_len % batch_size
#    if num_unbatched != 0:
#        paddings = [[0, num_unbatched], [0, 0], [0, 0]]
#        inp = tf.pad(inp, paddings, "CONSTANT")
#    return inp

# if __name__=="__main__":
