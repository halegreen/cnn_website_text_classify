import tensorflow as tf
import data_helper
import word2vec_helpers
from text_cnn import TextCNN
import pandas as pd
import numpy as np
import re
import jieba

#model hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 80,'dimensionality of characters')

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def input_valid_data(input_file):
    df = pd.read_excel(input_file)
    contents = df['web_content'].values
    valid_data = []
    for content in contents:
        content = data_helper.clean_str(find_chinese(content))
        content = data_helper.seperate_line(content)
        valid_data.append(content)
    return valid_data

def find_chinese(content):
    pattern = re.compile(u"([\u4e00-\u9fa5]+)")
    content = pattern.findall(content)
    return ' '.join(content)

def validData2vec(sentences):
    print 'Word embedding...'
    all_vectors = word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, 
            file_to_load = '/home/WXX/WebClassify/cnn_website_text_classify/runs/1503023156/trained_word2vec.model') 
    x_valid = np.array(all_vectors)
    return x_valid

def check_valid_data(data):
    document_num = len(data)
    max_document_length = len(data[0])
    embedding_dim = len(data[0][0])
    cnt0 = 0
    cnt1 = 0
    for doc in data:
        if len(doc) != max_document_length:
            cnt0 += 1
        for vec in doc:
            if len(vec) != embedding_dim:
                cnt1 += 1
    print 'sentence size inconsistent num : ' , cnt0    # cnt0 != 0, this dim's size is inconsistent! but why?
    print 'embedding vec size inconsistent num: ' , cnt1

def check_padding_sentences(input_sentences, x_raw):
    valid_sentences = []
    new_x_raw = []
    valid_length = len(input_sentences[0])
    print 'checking padding sentences..., valid length : ', valid_length 
    for sentence in input_sentences:
        if len(sentence) == valid_length:
            valid_sentences.append(sentence)
            new_x_raw.append(x_raw[input_sentences.index(sentence)])
    return (valid_sentences, new_x_raw) 

if __name__ == '__main__':
    #load params
    params_file = '/home/WXX/WebClassify/cnn_website_text_classify/runs/1503023156/training_params.pickle'
    params = data_helper.loadDict(params_file)
    num_labels = int(params['num_labels'])
    max_document_length  = int(params['max_document_length'])
    #input valid data and 2vec
    print '\nInput valid data...\n'
    valid_data = input_valid_data('data/topdomain20170801_crawler_new.xlsx')[0:100]
    print 'Padding sentenses ...'
    sentences, max_document_length = data_helper.padding_sentences(valid_data, '<PADDING>', padding_sentence_length = max_document_length)
    print 'max document length: ', max_document_length
    sentences =  check_padding_sentences(sentences)
    
    x_valid = validData2vec(sentences)
    print ('x_valid.shape = {}'.format(x_valid.shape))
    check_valid_data(x_valid)
