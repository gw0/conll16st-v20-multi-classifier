#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Discourse relation sense classification (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import argparse
import logging
import os
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.callbacks import Callback, ModelCheckpoint

from conll16st.load import load_all
from tasks.common import conv_window_to_offsets, save_to_pkl, load_from_pkl
from tasks.words import build_words2id
from tasks.pos_tags import build_pos_tags2id
from tasks.rel_types import build_rel_types2id, decode_x_rel_types
from tasks.rel_senses import build_rel_senses2id, decode_x_rel_senses
from tasks.rel_marking import build_rel_marking2id
from model import build_model, relation_sample, batch_generator


# logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

# attach debugger
def debugger(type, value, tb):
    import traceback, pdb
    traceback.print_exception(type, value, tb)
    pdb.pm()
import sys
sys.excepthook = debugger

# parse arguments
argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
argp.add_argument('experiment_dir',
    help="directory for storing trained model and other resources")
argp.add_argument('train_dir',
    help="CoNLL15st dataset directory for training")
argp.add_argument('valid_dir',
    help="CoNLL15st dataset directory for validation")
argp.add_argument('test_dir',
    help="CoNLL15st dataset directory for testing (only 'parses.json')")
argp.add_argument('output_dir',
    help="output directory for system predictions (in 'output.json')")
argp.add_argument('--clean', action='store_true',
    help="clean previous experiment")
args = argp.parse_args()

# defaults
epochs = 10000
batch_size = 10

word_crop = 100  #= max([ len(s) for s in train_words ])
embedding_dim = 40  #100
words2id_size = 50000  #= None is computed
skipgram_window_size = 4
skipgram_negative_samples = 0  #skipgram_window_size
skipgram_offsets = conv_window_to_offsets(skipgram_window_size, skipgram_negative_samples, word_crop)
filter_types = None  #["Explicit"]
filter_senses = None  #["Contingency.Condition"]
max_len = word_crop + max(abs(min(skipgram_offsets)), abs(max(skipgram_offsets)))

log.info("configuration ({})".format(args.experiment_dir))
for var in ['args.experiment_dir', 'args.train_dir', 'args.valid_dir', 'args.test_dir', 'args.output_dir', 'K._config', 'os.getenv("THEANO_FLAGS")', 'epochs', 'batch_size', 'word_crop', 'embedding_dim', 'words2id_size', 'skipgram_window_size', 'skipgram_negative_samples', 'skipgram_offsets', 'filter_types', 'filter_senses', 'max_len']:
    log.info("  {}: {}".format(var, eval(var)))

# experiment files
if args.clean and os.path.isdir(args.experiment_dir):
    import shutil
    shutil.rmtree(args.experiment_dir)
if not os.path.isdir(args.experiment_dir):
    os.makedirs(args.experiment_dir)
words2id_pkl = "{}/words2id.pkl".format(args.experiment_dir)
pos_tags2id_pkl = "{}/pos_tags2id.pkl".format(args.experiment_dir)
rel_types2id_pkl = "{}/rel_types2id.pkl".format(args.experiment_dir)
rel_senses2id_pkl = "{}/rel_senses2id.pkl".format(args.experiment_dir)
rel_marking2id_pkl = "{}/rel_marking2id.pkl".format(args.experiment_dir)
model_yaml = "{}/model.yaml".format(args.experiment_dir)
model_png = "{}/model.png".format(args.experiment_dir)
stats_csv = "{}/stats.csv".format(args.experiment_dir)
weights_hdf5 = "{}/weights.hdf5".format(args.experiment_dir)

# load datasets
log.info("load dataset for training ({})".format(args.train_dir))
train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, train_relations_gold = load_all(args.train_dir, filter_types=filter_types, filter_senses=filter_senses)
log.info("  doc_ids: {}, words: {}, rel_ids: {}, relation tokens: {}".format(len(train_doc_ids), sum([ len(s) for s in train_words.itervalues() ]), len(train_rel_ids), sum([ train_rel_parts[rel_id]['TokenCount'] for rel_id in train_rel_parts ])))
if not train_doc_ids:
    raise IOError("Failed to load dataset!")

log.info("load dataset for validation ({})".format(args.valid_dir))
valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, valid_relations_gold = load_all(args.valid_dir, filter_types=filter_types, filter_senses=filter_senses)
log.info("  doc_ids: {}, words: {}, rel_ids: {}, relation tokens: {}".format(len(valid_doc_ids), sum([ len(s) for s in valid_words.itervalues() ]), len(valid_rel_ids), sum([ valid_rel_parts[rel_id]['TokenCount'] for rel_id in valid_rel_parts ])))
if not valid_doc_ids:
    raise IOError("Failed to load dataset!")

log.info("load dataset for testing ({})".format(args.test_dir))
test_doc_ids, test_words, test_word_metas, test_pos_tags, test_dependencies, test_parsetrees, test_rel_ids, test_rel_parts, test_rel_types, test_rel_senses, test_relations_gold = load_all(args.test_dir, filter_types=filter_types, filter_senses=filter_senses)
log.info("  doc_ids: {}, words: {}, rel_ids: {}, relation tokens: {}".format(len(test_doc_ids), sum([ len(s) for s in test_words.itervalues() ]), len(test_rel_ids), sum([ test_rel_parts[rel_id]['TokenCount'] for rel_id in test_rel_parts ])))
if not test_doc_ids:
    raise IOError("Failed to load dataset!")

# build indexes
if not all([ os.path.isfile(pkl) for pkl in [words2id_pkl, pos_tags2id_pkl, rel_types2id_pkl] ]):
    log.info("build indexes")
    words2id, words2id_size = save_to_pkl(words2id_pkl, build_words2id(train_words, max_size=words2id_size))
    pos_tags2id, pos_tags2id_size = save_to_pkl(pos_tags2id_pkl, build_pos_tags2id(train_pos_tags))
    rel_types2id, rel_types2id_size = save_to_pkl(rel_types2id_pkl, build_rel_types2id(train_rel_types))
    rel_senses2id, rel_senses2id_size = save_to_pkl(rel_senses2id_pkl, build_rel_senses2id(train_rel_senses))
    rel_marking2id, rel_marking2id_size = save_to_pkl(rel_marking2id_pkl, build_rel_marking2id(mode='IO'))
else:
    log.info("load previous indexes ({})".format(args.experiment_dir))
    words2id, words2id_size = load_from_pkl(words2id_pkl)
    pos_tags2id, pos_tags2id_size = load_from_pkl(pos_tags2id_pkl)
    rel_types2id, rel_types2id_size = load_from_pkl(rel_types2id_pkl)
    rel_senses2id, rel_senses2id_size = load_from_pkl(rel_senses2id_pkl)
    rel_marking2id, rel_marking2id_size = load_from_pkl(rel_marking2id_pkl)
log.info("  words2id: {}, pos_tags2id: {}, rel_types2id: {}, rel_senses2id: {}, rel_marking2id: {}".format(words2id_size, pos_tags2id_size, rel_types2id_size, rel_senses2id_size, rel_marking2id_size))

# build model
log.info("build model")
model = build_model(max_len, embedding_dim, words2id_size, pos_tags2id_size, rel_types2id_size, rel_senses2id_size, rel_marking2id_size)

# plot model
with open(model_yaml, 'w') as f:
    model.to_yaml(stream=f)
plot(model, model_png)

# initialize model
if not os.path.isfile(weights_hdf5):
    log.info("initialize new model")
else:
    log.info("load previous model ({})".format(args.experiment_dir))
    model.load_weights(weights_hdf5)

#XXX
import numpy as np
class SenseValidation(Callback):
    """Discourse relation sense validation."""

    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        rel_types_matches = 0
        rel_senses_matches = 0
        for rel_id in valid_rel_ids:
            # predict for each relation separately
            x1_words_pad, x1_words_rand, x1_skipgram, x1_pos_tags, x1_rel_types, x1_rel_senses, x1_rel_focus, token_start, token_end = relation_sample(rel_id, word_crop, max_len, valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, words2id, words2id_size, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size)
            y = self.model.predict({
                'x_words_pad': np.asarray([x1_words_pad], dtype=np.int),
                #'x_words_rand': np.asarray([x1_words_rand], dtype=np.int),
                'x_rel_focus': np.asarray([x1_rel_focus], dtype=np.float32),
                #'x_skipgram': np.asarray([x1_skipgram], dtype=np.float32),
                #'x_pos_tags': np.asarray([x1_pos_tags], dtype=np.float32),
                #'x_rel_types': np.asarray([x1_rel_types], dtype=np.float32),
                #'x_rel_senses': np.asarray([x1_rel_senses], dtype=np.float32),
            })

            rel_type, rel_type_totals = decode_x_rel_types(y['x_rel_types'][0], range(token_start, token_end), valid_rel_parts[rel_id], rel_types2id, rel_types2id_size)
            if rel_type == valid_rel_types[rel_id]:
                rel_types_matches += 1

            #rel_sense, rel_sense_totals = decode_x_rel_senses(y['x_rel_senses'][0], range(token_start, token_end), valid_rel_parts[rel_id], rel_senses2id, rel_senses2id_size)
            #if rel_sense == valid_rel_senses[rel_id]:
            #    rel_senses_matches += 1

            #if valid_rel_types[rel_id] != rel_type:
            #    print rel_id, valid_rel_types[rel_id], rel_type, rel_type_totals
                # print (np.repeat(x1_rel_focus, 5).reshape(102,5)) * y['x_rel_types'][0]
                # print (1 - np.repeat(x1_rel_focus, 5).reshape(102,5)) * y['x_rel_types'][0]
                # print x1_rel_types

        print len(valid_rel_ids), rel_types_matches, rel_senses_matches

#XXX
rel_types2id[None] = []
np.set_printoptions(precision=2, suppress=True)

# train model
train_iter = batch_generator(word_crop, max_len, batch_size, train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, words2id, words2id_size, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size)
callbacks = [
    #XXX:CSVHistory(stats_csv),
    ModelCheckpoint(filepath=weights_hdf5, monitor='avg_loss', mode='min', save_best_only=True),
    SenseValidation(model),
]
model.fit_generator(train_iter, nb_epoch=epochs, samples_per_epoch=len(train_rel_ids), callbacks=callbacks)

# predict model
# log.info("predict model")

