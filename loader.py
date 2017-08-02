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
        string = f.read()
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
        #Add a full stop to 1-word samples so that tf's CRF API can run over a sequence.
        #CRF API throws an error for seqlen==1
        if len(sample.split('\n'))==1:
            input_x[-1].append('.')
            input_y[-1].append('.')
    return input_x, input_y


target_to_source_mappings = {
    "MC": "CD",
    "DD": "DT",
    "II": "IN",
    "JJS": "JJT",
    "RR": "RB",
    "RRR": "RBR",
    "RRT": "RBS",
    "VM": "MD",
    "GE": "POS",
    "DB": "PDT",
    "PNR": "WDT",
    "PN": "PRP",
    "PNG": "PRP$",
    "PND": "DT",
    "VVZ": "VBZ",
    "VDZ": "VBZ",
    "VHZ": "VBZ",
    "VBI": "VB",
    "VVI": "VB",
    "VHI": "VB",
    "VHB": "VBP",
    "VVB": "VBP",
    "VDN": "VBN",
    "VVN": "VBN",
    "VDD": "VBD",
    "VHD": "VBD",
    "VVD": "VBD",
    "VHG": "VBG",
    "VVG": "VBG",
    "VVGN": "NN",
    "VVGJ": "JJ",
    "VVNJ": "JJ"
    }

def map_tags(data):
    #Replace all tags of format <<tag+>> by <<tag>>.
    data = re.sub(r'(_[A-Z]+)\+', r'\g<1>', data)
    #Change more_RR/less_RR to more_RBR/less_RBR
    data = re.sub(r'(more|less_)RR ', r'\g<1>RBR ', data)
    #Change more/less_DD with more/less_JJR
    data = re.sub(r'(more|less_)DD ', r'\g<1>JJR ', data)
    #Replace most_PND with most_JJS
    data = re.sub(r'(more_)PND ', r'\g<1>JJS ', data)
    #Replace few_PND with few_JJ
    data = re.sub(r'(few|same_)PND ', r'\g<1>JJ ', data)
    #Replace be_VBB with be_VB
    data = re.sub(r'(be_)VBB ', r'\g<1>VB ', data)
    # Replace are_VBB with are_VBP
    data = re.sub(r'(are_)VBB ', r'\g<1>VBP ', data)
    #Replace to_TO do_VDB with do_VB
    data = re.sub(r'(to_TO do_)VDB ', r'\g<1>VB ', data)
    #Replace remaining do_VDB with do_VB
    data = re.sub(r'(do_)VDB ', r'\g<1>VBP ', data)
    #Replace direct tag mapping schemes from the dictionary
    for ttag, stag in target_to_source_mappings.items():
        data = re.sub(ttag+" ", stag+" ", data)
    return data


def prepare_medpost_input():
    dir = config.medpost_train_datadir
    input_x = []
    input_y = []
    for file in os.listdir(dir):
        with open(dir+file, 'r') as f:
            data = f.read()
            data = map_tags(data)
            lines = data.split('\n')
            lines.pop()
        if (len(lines)%2)!=0:
            raise Exception("File has odd number of lines. Check file %s", file)
        for ind in range(len(lines)/2):
            seq = lines[ind*2+1]
            if seq=='':
                continue
            input_x.append([])
            input_y.append([])
            tokens = seq.split()
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
    saver = tf.train.import_meta_graph("source_model.meta")
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


