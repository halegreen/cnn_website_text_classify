# encoding=utf8
import re
import jieba
import itertools
import os
from collections import defaultdict
import numpy as np
import io
import codecs
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


'''
input text data and it's label
'''

def file_helper(input_file):
    lines = list(open(input_file, 'r').readlines())
    label2cont = defaultdict(list)
    for line in lines:
        idx = line.find(':')
        if idx == -1:
            continue
        label2cont[str(line[ : idx]).strip()].append(str(line[idx + 1 :]).strip())
    for key, value in label2cont.iteritems():
        with open('input_data/'+ str(key) + ".txt", 'w') as f:
            for line in label2cont[key]:
                f.write((line + '\n'))


def load_data_and_label(input_file):
    classNames = os.listdir(input_file)
    x_train = []
    classes = []
    labels = []
    for c in classNames:
        tmp = []
        cont = []
        with  codecs.open(input_file + '/' + c, 'r', encoding='utf-8', errors='ignore') as f:
            tmp = [line.strip() for line in f.readlines()]
        for t in tmp:
            #print seperate_line(clean_str(t))
            cont.append(seperate_line(clean_str(t)))
        classes.append(cont)
    #genarate classee
    for c in classes:
        x_train += c
   
    idx = 0
    for i in range(len(classes)):
        labelvec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labelvec[idx] = 1
        idx += 1
        tmplabel = [labelvec for _ in classes[i]]
        labels.append(tmplabel)
    #combine label
    y_train = np.concatenate(labels, 0)

    return [x_train, y_train, label2str]

def label2str(input_file):
    classNames = os.listdir(input_file)
    label2str = {}
    i = 0
    for c in classNames:
        label2str[i] = c.split('.')[0]
        i += 1
    return label2str

def clean_str(string):
    #string = re.sub(ur"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    #string = re.sub(r"\s{2,}", " ", string)
    string = re.sub('\s+', " ", string)
    r1 = u'[A-Za-z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    string = re.sub(r1, ' ', string)
    return string.strip()

def seperate_line(line):
    line = jieba.cut(line)
    return ''.join([word + " " for word in line])

def batch_iter(data, batch_size, epoch_num, shuffle = True):
    data = np.array(data)
    data_size = len(data)
    batch_num_per_epoch = int((data_size - 1 / batch_size)) + 1
    for epoch in range(epoch_num):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
    for batch_num in range(batch_num_per_epoch):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_idx : end_idx]

def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
    sentences = [sentences.split() for sentences in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)

def saveDict(input_dict, output_file):
    with open(output_file, 'w') as f:
        pickle.dump(input_dict, f)

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'r') as f:
        output_dict = pickle.load(f)
    return output_dict

if __name__ == '__main__':
    #file_helper("trainning_data/40_million_training_data_12/big_type_str_12.txt")
    #x_text, y = load_data_and_label("input_data/")
    #print "len of x_train is: ", len(x_text)
    #print "len of y_train is: ", y.shape
    #sentences, max_document_length = padding_sentences(x_text[0:100], '<PADDING>')
    #print "max document length = ", max_document_length
    label2str = label2str('input_data/')
    for k, v in label2str.items():
        print 'label : %d' % k, ' ', 'class : %s'  % v  
    # sentences = [sentences.split( ) for sentences in x_text[0:5000]]
