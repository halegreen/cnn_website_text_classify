#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
import word2vec_helpers
from text_cnn import TextCNN
import csv
import eval_helper

# Parameters
# ==================================================

# Eval Parameters
#tf.flags.DEFINE_integer("batch_size", 20, "Batch Size (default: 20)")
#tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
#tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_integer("num_labels", 12, "Number of labels for data. (default: 12)")
#tf.flags.DEFINE_integer("embedding_dim", 80, "Dimensionality of character embedding (default: 80)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# validate
# ==================================================

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file
trained_word2vec_model_file = os.path.join(FLAGS.checkpoint_dir, "..", "trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# Load params
params = data_helper.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])
#max_document_length = 944

# Load data
if FLAGS.eval_train:
    x_raw = eval_helper.input_valid_data('data/topdomain20170801_crawler_new.xlsx')[0:200]
    print 'raw data length : %d' % len(x_raw)
    y_test = None
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

label2str = data_helper.label2str('input_data/')

# Get Embedding vector x_test
print 'Padding sentence...'
sentences, max_document_length = data_helper.padding_sentences(x_raw, '<PADDING>', padding_sentence_length = max_document_length)
print 'sentences length : %d , max_document_length : %d' % (len(sentences),  max_document_length)
sentences, new_x_raw = eval_helper.check_padding_sentences(sentences, x_raw)

all_vectors = word2vec_helpers.embedding_sentences(sentences,embedding_size = 128, file_to_load = trained_word2vec_model_file)
print 'all_vectors length: %d' %  len(all_vectors[0])
x_test = np.array(all_vectors)
print("x_test.shape = {}".format(x_test.shape))
print 'x_test_shape: '  , x_test.shape, " ", len(x_test) ," " , len(x_test[0]) ," " , len(x_test[0][0])
print 'list x_test ', len(list(x_test))

# Evaluation
# ==================================================
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

for p in all_predictions:
    p = label2str[p]

print ' prediction data num: ', len(all_predictions)

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array([text.encode('utf-8') for text in new_x_raw]), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
