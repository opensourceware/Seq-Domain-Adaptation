import os, re
import nltk
# from nltk.parse import stanford
# from nltk.tokenize import RegexpTokenizer
# import gensim
import numpy as np
import tensorflow as tf
import config
import utils


def load_emb(word_to_index, weights_loc):
    """
    Returns a numpy array of weights (vocab_size, word_dim)
    """
    with open(weights_loc, 'r') as file:
        weights = file.read().split('\n')
        if weights[-1] == '':
            weights.pop()
        weights = np.array([np.array(weights[index].split()) for index in word_to_index.values()])
        weights = weights.astype(np.float)
        mean = np.mean(weights, axis=0)
        std = np.std(weights, axis=0)
        oov = np.random.normal(mean, std)
        weights = np.vstack((weights, oov))
    return weights


def build_vocab():
    """
    Builds vocabulary from training data to reduce embedding matrix size.
    Returns: A set of words in the training set.
    """
    vocab = []
    with open(config.datadir+config.train, 'r') as f:
        string = f.read().lower()
        words = nltk.word_tokenize(string)
        vocab += words
    vocab = set(vocab)
    return vocab


def prepare_input(file):
    """
    Builds input and label batches from raw text files.
    Arg: Filename with raw data - 'train.txt', 'val.txt', 'test.txt'
    Returns: A list of batches of input sequence and tag sequence
    """
    input_x = []
    input_y = []
    with open(file, 'r') as f:
        samples = f.read().split('\n\n')
    for sample in samples:
        if sample=='':
            continue
        input_x.append([])
        input_y.append([])
        for word in sample.split('\n'):
            input_x[-1].append(word.split()[0])
            input_y[-1].append(word.split()[1])
    return input_x, input_y


def prepare_medpost_input(dir):
    input_x = []
    input_y = []
    for file in os.listdir(dir):
        with open(file, 'r') as f:
            lines = f.read().split('\n')
        if (len(lines)%2)!=0:
            raise Exception("File has odd number of lines. Check file %s", file)
        sample_index = range(len(lines)/2)
        for ind in sample_index:
            seq = lines[ind*2+1]
            if seq=='':
                continue
            input_x.append([])
            input_y.append([])
            tokens = nltk.word_tokenize(seq)
            for token in tokens:
                input_x[-1].append(token.split('_')[0])
                input_y[-1].append(token.split('_')[1])
    return input_x, input_y


def load_and_save_weights():
    vocab = build_vocab()
    word_to_index, _ = utils.word_to_index(vocab)
    weights = load_emb(word_to_index, config.PRETRAINED_VECTORS)
    np.savetxt('emb.mat', weights)


def reload_smodel(sess):
    saver = tf.train.import_meta_graph("model.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))
    graph = tf.get_default_graph()
    return graph

def save_smodel(sess):
    saver = tf.train.Saver()
    saver.save(sess, "./source_model")

if __name__ == "__main__":
    #vocab = build_vocab()
    #word_to_index = utils.word_to_index(vocab)
    #weights = load_emb(word_to_index, config.PRETRAINED_VECTORS)
    input_x, input_y = prepare_input(config.datadir + config.train)


