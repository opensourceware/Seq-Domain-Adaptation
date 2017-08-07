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


def word_to_index_glove(vocab, glove_emb_path):
    """
    :param glove_emb_path: file containing weights
    :return:
    :param: vocab - vocabulary consists of 2.2 million words in the glove dictionary
    :param: weights - weight matrix of (vocab_size, word_dim)
    """
    pad = np.zeros(shape=300, dtype='float32')
    unk = np.random.normal(0.001, 0.01, 300)

    with open(glove_emb_path, 'r') as file:
        glove_weights = file.read().split('\n')
        if glove_weights[-1] == '':
            glove_weights.pop()
    glove_vocab = [w.split()[0] for w in glove_weights]
    word_to_id = {}
    word_to_id['PAD'] = 0
    index = 1
    weights = []
    for word in vocab:
        if (word in glove_vocab):
            ind = glove_vocab.index(word)
            glove_weight = np.array(glove_weights[ind].split()[1:]).astype(np.float32)
            weights.append(glove_weight)
            word_to_id[word] = index
            index += 1
    word_to_id['UNK'] = index
    weights = np.array(weights).astype(np.float)
    weights = np.vstack((pad, weights, unk))
    return weights, word_to_id


def word_to_index_word2vec(vocab, word2vec_model):
    emb_size = word2vec_model['the'].shape[0]
    print emb_size
    pad = np.zeros(shape=emb_size, dtype='float32')
    unk = np.random.normal(0.001, 0.01, emb_size)

    word_to_id = {}
    word_to_id['PAD'] = 0
    index = 1
    weights = []
    for word in vocab:
        if word in word2vec_model:
            weights.append(word2vec_model[word])
            word_to_id[word] = index
            index += 1
    word_to_id['UNK'] = index
    weights = np.array(weights).astype(np.float32)
    weights = np.vstack((pad, weights, unk))
    return weights, word_to_id


def word_to_index(vocab, word2vec_model, glove_emb_path):
    pad = np.zeros(shape=600, dtype='float32')
    unk = np.random.normal(0.001, 0.01, 600)
    with open(glove_emb_path, 'r') as file:
        glove_weights = file.read().split('\n')
        if glove_weights[-1] == '':
            glove_weights.pop()
    glove_vocab = [w.split()[0] for w in glove_weights]
    word_to_id = {}
    word_to_id['PAD'] = 0
    index = 1
    weights = []
    for word in vocab:
        if (word in glove_vocab) and (word in word2vec_model):
            ind = glove_vocab.index(word)
            glove_weight = np.array(glove_weights[ind].split()[1:]).astype(np.float32)
            word2vec_weight = word2vec_model[word]
        elif (word in glove_vocab) and (word not in word2vec_model):
            ind = glove_vocab.index(word)
            glove_weight = np.array(glove_weights[ind].split()[1:]).astype(np.float32)
            word2vec_weight = unk[300:]
        elif (word not in glove_vocab) and (word in word2vec_model):
            glove_weight = unk[:300]
            word2vec_weight = word2vec_model[word]
        else:
            continue
        weights.append(np.concatenate((glove_weight, word2vec_weight)))
        word_to_id[word] = index
        index += 1
    word_to_id['UNK'] = index
    weights = np.array(weights).astype(np.float)
    weights = np.vstack((pad, weights, unk))
    return weights, word_to_id


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


def convert_to_char_emb(input_x):
    """
    Create indexes of characters in the string. To pass an input with consistent length,
    words are paded with zeros to match the length of longest word in the sentence.
    :param input_x:
    :return:
    char_emb: one-hot indexed of characters in the words
    char_to_id: a dict mapping characters to respective id
    char_seqlen: list of actual length of words in sequences. The shape of list is
    [num_sequences,len(sequence)]. This list doesn't have a consistent shape in the 2nd dimension which is
    not a problem since batch_size=1. Passed to brnn.
    TODO: Design for batch_size>1
    """
    char_emb = []
    char_to_id = {}
    ind = 0
    max_char_len = 0
    for seq in input_x:
        for word in seq:
            for char in word:
                if char not in char_to_id:
                    char_to_id[char] = ind
                    ind+=1
    char_emb_size = len(char_to_id)
    char_seqlen = []
    for seq in input_x:
        char_emb.append([])
        char_seqlen.append([len(x) for x in seq])
        max_word_len = max(char_seqlen[-1])
        for word in seq:
            if len(word) > max_word_len:
                max_word_len = len(word)
            char_emb[-1].append([])
            for char in word:
                emb = [0] * char_emb_size
                emb[char_to_id[char]] = 1
                char_emb[-1][-1].append(emb)
            if len(char_emb[-1][-1])<max_word_len:
                padding = [[0]*char_emb_size]*(max_word_len-len(char_emb[-1][-1]))
                char_emb[-1][-1]+=padding
    return char_emb, char_to_id, char_seqlen


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
