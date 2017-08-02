# import nltk, os
# from nltk.parse import stanford
# import gensim
import config
import numpy as np
import tensorflow as tf
import sklearn


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
    """
    Converts words to their word IDs.
    :param input_x: a list of word sequences
    :param word_to_id: a dictionary mapping words to corresponding ids
    :return:
    :param seqlen: a list of lenghts of sequences
    :param Idx: mapped list of words sequences
    """
    seqlen = []
    Idx = []
    for sent in input_x:
        sentIdx = []
        for word in sent:
            try:
                sentIdx.append(word_to_id[word])
            except KeyError:
                sentIdx.append(word_to_id['UNK'])
        seqlen.append(len(sent))
        Idx.append(sentIdx)
    return seqlen, Idx


def create_and_convert_tag_to_id(tags):
    """
    Used during training time when a disctionary mapping tags to id is created
    and the train tags are converted to their ids simultaneously.
    :param tags:
    :return:
    """
    tag_to_id = {}
    i = 0
    Idx = []
    for sent in tags:
        sentIdx = []
        for tag in sent:
            if tag not in tag_to_id:
                tag_to_id[tag] = i
                i += 1
            sentIdx.append(tag_to_id[tag])
        Idx.append(sentIdx)
    return Idx, tag_to_id


def convert_tag_to_id(tag_to_id, tags):
    """
    Uses tag_to_id mapping to convert dev or test set data into their corresponding ids.
    :param tags:
    :return:
    """
    Idx = []
    for sent in tags:
        sentIdx = []
        for tag in sent:
            if tag not in tag_to_id:
                raise Exception("Error dev/test time tag %s not seen during training "+ tag)
            sentIdx.append(tag_to_id[tag])
        Idx.append(sentIdx)
    return Idx


def create_batches(input_x, seqlen, input_y=None):
    """
    Implements batch creation by bucketing of same length sequences.
    :param input_x: List of indices of words in the sequence
    :param input_y: List of indices of tags in the sequence
    :param seqlen:
    :returns
    :param seqbatch: A list of sequence length of each bucket
    :param batches: A list of buckets of input/output pairs.
    """
    if input_y is None:
        inp = input_x
        inp = sorted(inp, key=lambda x: len(x))
    else:
        inp = []
        for num, item in enumerate(input_x):
            inp.append([item, input_y[num]])
        inp = sorted(inp, key=lambda x:len(x[0]))
    seqlen = sorted(seqlen)

    batches = []
    seqbatch = []
    prev_len = 1
    for num, seq in enumerate(inp):
        if (num==0) or ((len(batches[-1])%config.batch_size)==0):
            prev_len = seqlen[num]
            batches.append([seq])
            seqbatch.append([seqlen[num]])
        elif seqlen[num]==prev_len:
            batches[-1].append(seq)
            seqbatch[-1].append(seqlen[num])
        else:
            batches.append([seq])
            seqbatch.append([seqlen[num]])
            prev_len = seqlen[num]
    return seqbatch, batches


def pad_last_batch(inp, seqlen, maxseqlen, inp_len, batch_size):
    num_unbatched = inp_len % batch_size
    inp += [[[0]*maxseqlen, [0]*maxseqlen]]*(batch_size-num_unbatched)
    seqlen += [0]*(batch_size-num_unbatched)
    return inp, seqlen


def get_batch(data):
    num_buckets = len(data)
    randint = np.random.randint(num_buckets)
    return randint, data[randint]


def eval(predictions, true_labels, tag_to_id):
    class_wise_f1 = sklearn.metrics.f1_score(predictions, true_labels, average=None)
    macro_avg = sklearn.metrics.f1_score(predictions, true_labels, average="macro")
    print "Class wise F1 score is as follows:"
    for tag, id in tag_to_id.items():
        try:
            print tag + "\t:\t" + str(class_wise_f1[id])
        except IndexError:
            continue
    print "\n\n" + "Macro Avg F1 score is "+str(macro_avg)


