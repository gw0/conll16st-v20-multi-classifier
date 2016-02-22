#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Discourse relation sense classification (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import argparse
import codecs
import logging
import os
import sys
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint, EarlyStopping

from conll16st.load import load_all
from tasks.common import conv_window_to_offsets, save_to_pkl, load_from_pkl
from tasks.words import build_words2id
from tasks.pos_tags import build_pos_tags2id
from tasks.rel_types import build_rel_types2id
from tasks.rel_senses import build_rel_senses2id
from tasks.rel_marking import build_rel_marking2id
from model import build_model, batch_generator, RelationMetrics


# logging
class Tee(object):
    """For redirecting output to console and log files."""

    def __init__(self, direct=[], files=[]):
        self.direct = list(direct)
        self.files = list(files)
        self.buf = ""

    def write(self, obj):
        # direct output
        for f in self.direct:
            f.write(obj)

        # buffered line output to files
        self.buf += obj
        line = ""
        for line in self.buf.splitlines(True):
            if line.endswith("\n"):  # write only whole lines
                for f in self.files:
                    f.write(line)
                line = ""
        self.buf = line  # preserve last unflushed line

    def flush(self) :
        for f in self.direct + self.files:
            f.flush()

sys.stdout = Tee([sys.stdout])
sys.stderr = Tee([sys.stderr])

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

# attach debugger
def debugger(type, value, tb):
    import traceback, pdb
    traceback.print_exception(type, value, tb)
    pdb.pm()
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
epochs = 1000
epochs_patience = 20
batch_size = 32
batch_size_valid = 512
snapshot_size = 2000

word_crop = 100  #= max([ len(s) for s in train_words ])
embedding_dim = 40  #= 40
dropout_p = 0.5  #= 0.5
words2id_size = 50000  #= None is computed
skipgram_window_size = 10
skipgram_negative_samples = skipgram_window_size  #= skipgram_window_size
skipgram_offsets = conv_window_to_offsets(skipgram_window_size, skipgram_negative_samples, word_crop)
filter_types = None  #["Explicit"]
filter_senses = None  #["Contingency.Condition"]
max_len = word_crop + max(abs(min(skipgram_offsets)), abs(max(skipgram_offsets)))

# files
console_log = "{}/console.log".format(args.experiment_dir)
model_yaml = "{}/model.yaml".format(args.experiment_dir)
model_png = "{}/model.png".format(args.experiment_dir)
metrics_csv = "{}/metrics.csv".format(args.experiment_dir)
metrics_png = "{}/metrics.png".format(args.experiment_dir)
weights_hdf5 = "{}/weights.hdf5".format(args.experiment_dir)
words2id_pkl = "{}/words2id.pkl".format(args.experiment_dir)
pos_tags2id_pkl = "{}/pos_tags2id.pkl".format(args.experiment_dir)
rel_types2id_pkl = "{}/rel_types2id.pkl".format(args.experiment_dir)
rel_senses2id_pkl = "{}/rel_senses2id.pkl".format(args.experiment_dir)
rel_marking2id_pkl = "{}/rel_marking2id.pkl".format(args.experiment_dir)

# initialize experiment
if args.clean and os.path.isdir(args.experiment_dir):
    import shutil
    shutil.rmtree(args.experiment_dir)
if not os.path.isdir(args.experiment_dir):
    os.makedirs(args.experiment_dir)
f_log = codecs.open(console_log, mode='a', encoding='utf8')
try:
    sys.stdout.files.append(f_log)
    sys.stderr.files.append(f_log)
except AttributeError:
    f_log.close()

log.info("configuration ({})".format(args.experiment_dir))
for var in ['args.experiment_dir', 'args.train_dir', 'args.valid_dir', 'args.test_dir', 'args.output_dir', 'K._config', 'os.getenv("THEANO_FLAGS")', 'epochs', 'epochs_patience', 'batch_size', 'batch_size_valid', 'word_crop', 'embedding_dim', 'dropout_p', 'words2id_size', 'skipgram_window_size', 'skipgram_negative_samples', 'skipgram_offsets', 'filter_types', 'filter_senses', 'max_len']:
    log.info("  {}: {}".format(var, eval(var)))

# load datasets
log.info("load dataset for training ({})".format(args.train_dir))
train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, train_relations_gold = load_all(args.train_dir, filter_types=filter_types, filter_senses=filter_senses)
log.info("  doc_ids: {}, words: {}, rel_ids: {}, relation tokens: {}".format(len(train_doc_ids), sum([ len(s) for s in train_words.itervalues() ]), len(train_rel_ids), sum([ train_rel_parts[rel_id]['TokenCount'] for rel_id in train_rel_parts ])))
if not train_doc_ids:
    raise IOError("Failed to load dataset!")

log.info("load dataset for validation ({})".format(args.valid_dir))
if args.valid_dir == args.train_dir:
    valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, valid_relations_gold = train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, train_relations_gold
else:
    valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, valid_relations_gold = load_all(args.valid_dir, filter_types=filter_types, filter_senses=filter_senses)
log.info("  doc_ids: {}, words: {}, rel_ids: {}, relation tokens: {}".format(len(valid_doc_ids), sum([ len(s) for s in valid_words.itervalues() ]), len(valid_rel_ids), sum([ valid_rel_parts[rel_id]['TokenCount'] for rel_id in valid_rel_parts ])))
if not valid_doc_ids:
    raise IOError("Failed to load dataset!")

log.info("load dataset for testing ({})".format(args.test_dir))
if args.test_dir == args.valid_dir:
    test_doc_ids, test_words, test_word_metas, test_pos_tags, test_dependencies, test_parsetrees, test_rel_ids, test_rel_parts, test_rel_types, test_rel_senses, test_relations_gold = valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, valid_relations_gold
else:
    test_doc_ids, test_words, test_word_metas, test_pos_tags, test_dependencies, test_parsetrees, test_rel_ids, test_rel_parts, test_rel_types, test_rel_senses, test_relations_gold = load_all(args.test_dir, filter_types=filter_types, filter_senses=filter_senses)
log.info("  doc_ids: {}, doc words: {}, rel_ids: {}, rel tokens: {}".format(len(test_doc_ids), sum([ len(s) for s in test_words.itervalues() ]), len(test_rel_ids), sum([ test_rel_parts[rel_id]['TokenCount'] for rel_id in test_rel_parts ])))
if not test_doc_ids:
    raise IOError("Failed to load dataset!")

#XXX: release memory
train_relations_gold = None
train_dependencies = None
train_parsetrees = None

# build indexes
if not all([ os.path.isfile(pkl) for pkl in [words2id_pkl, pos_tags2id_pkl, rel_types2id_pkl] ]):
    log.info("build indexes")
    words2id, words2id_size = save_to_pkl(words2id_pkl, build_words2id(train_words, max_size=words2id_size))
    pos_tags2id, pos_tags2id_size = save_to_pkl(pos_tags2id_pkl, build_pos_tags2id(train_pos_tags))
    rel_types2id, rel_types2id_size = save_to_pkl(rel_types2id_pkl, build_rel_types2id(train_rel_types))
    rel_senses2id, rel_senses2id_size = save_to_pkl(rel_senses2id_pkl, build_rel_senses2id(train_rel_senses))
    rel_marking2id, rel_marking2id_size = save_to_pkl(rel_marking2id_pkl, build_rel_marking2id(mode='IO-part'))
else:
    log.info("load previous indexes ({})".format(args.experiment_dir))
    words2id, words2id_size = load_from_pkl(words2id_pkl)
    pos_tags2id, pos_tags2id_size = load_from_pkl(pos_tags2id_pkl)
    rel_types2id, rel_types2id_size = load_from_pkl(rel_types2id_pkl)
    rel_senses2id, rel_senses2id_size = load_from_pkl(rel_senses2id_pkl)
    rel_marking2id, rel_marking2id_size = load_from_pkl(rel_marking2id_pkl)
log.info("  words2id: {}, pos_tags2id: {}, rel_types2id: {}, rel_senses2id: {}, rel_marking2id: {}".format(words2id_size, pos_tags2id_size, rel_types2id_size, rel_senses2id_size, rel_marking2id_size))

#XXX
rel_types2id[None] = []

# build model
log.info("build model")
model = build_model(max_len, embedding_dim, dropout_p, words2id_size, skipgram_offsets, pos_tags2id_size, rel_types2id_size, rel_senses2id_size, rel_marking2id_size)

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
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, LogLocator, ScalarFormatter
from keras.callbacks import History

class CSVHistory(History):
    """Callback to store history metrics in CSV file.

    # Arguments
        metrics_csv: string, path to save the CSV file.
        csv_fields: list of fields/columns to store in CSV file.
        others: optional dictionary for fields with static values.
    """

    def __init__(self, metrics_csv, csv_fields=None, others=None):
        super(CSVHistory, self).__init__()
        if csv_fields is None:
            csv_fields = ['epoch', 'loss', 'val_loss']
        if others is None:
            others = {}

        self.metrics_csv = metrics_csv
        self.csv_fields = csv_fields
        self.others = others

    def on_train_begin(self, logs={}):
        super(CSVHistory, self).on_train_begin(logs=logs)
        try:
            self.load_csv()
        except IOError:
            pass

    def on_epoch_end(self, epoch, logs={}):
        super(CSVHistory, self).on_epoch_end(epoch, logs=logs)
        self.save_csv()

    def load_csv(self):
        f = open(self.metrics_csv, 'rb')
        freader = csv.DictReader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.history = {}
        for row in freader:
            for k, v in row.items():
                if k == 'epoch':  # load epoch numbers
                    self.epoch.append(v)
                elif k in self.others:  # skip other fields
                    pass
                else:  # load metrics
                    if k not in self.history:
                        self.history[k] = []
                    self.history[k].append(v)
        f.close()

    def save_csv(self):
        f = open(self.metrics_csv, 'wb')
        fwriter = csv.DictWriter(f, fieldnames=self.csv_fields, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fwriter.writeheader()
        for i in range(len(self.epoch)):
            row = {}
            for k in self.csv_fields:
                if k == 'epoch':  # save epoch numbers
                    row[k] = self.epoch[i]
                elif k in self.others:  # save other fields
                    row[k] = self.others[k]
                elif k in self.history:  # save metrics
                    row[k] = self.history[k][i]
            fwriter.writerow(row)
        f.close()

class PlotHistory(CSVHistory):
    """Callback to plot multiple metrics from history.

    # Arguments
        metrics_png: string, path to save the PNG image.
        png_fields: list of lists of fields/columns for each plot.
        metrics_csv: string, path to save the CSV file (from `CSVHistory`).
        csv_fields: list of fields/columns to store in CSV file (from `CSVHistory`).
        others: optional dictionary for fields with static values (from `CSVHistory`).
    """
    def __init__(self, metrics_png, png_fields, metrics_csv, csv_fields=None, others=None):
        super(PlotHistory, self).__init__(metrics_csv, csv_fields=csv_fields, others=others)

        self.metrics_png = metrics_png
        self.png_fields = png_fields

    def on_epoch_end(self, epoch, logs={}):
        super(PlotHistory, self).on_epoch_end(epoch, logs=logs)
        self.save_png()

    def save_png(self, title=None, crop_max=10.5, normalize_endswith='loss'):
        if title is None:
            title = ", ".join(self.others.values())

        fig, axarr = plt.subplots(len(self.png_fields), sharex=True)

        x = range(len(self.epoch))
        for fields, ax in zip(self.png_fields, axarr):
            for k in fields:
                if k in self.history:
                    vals = self.history[k]

                    # crop larger values
                    if crop_max:
                        vals = [ min(y, crop_max) for y in vals ]

                    # normalize to first value (for loss functions)
                    if k.endswith(normalize_endswith):
                        vals = [ (y / vals[0]) for y in vals ]

                    # plot
                    ax.plot(x, vals, label=k)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_minor_locator(MultipleLocator(base=10))
            ax.set_ylim(ymin=0.)
            ax.yaxis.set_minor_locator(MultipleLocator(base=0.05))
            #ax.set_yscale('log')
            #ax.set_ylim(ymin=0.1)
            #ax.yaxis.set_major_locator(LogLocator(base=2.))
            #ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True)
            ax.legend(fontsize = 'small', loc='upper left', bbox_to_anchor=(1., 1.))

        ax.set_xlabel('epochs')
        fig.suptitle(title, fontsize='large', y=1.)
        fig.tight_layout()
        fig.savefig(self.metrics_png, bbox_inches='tight', dpi=100)
        plt.close(fig)


from keras.callbacks import Callback

class EvaluateAllLosses(Callback):
    """Callback to evaluate all weighted losses individually."""

    def __init__(self, prefix, postfix, data, batch_size):
        super(EvaluateAllLosses, self).__init__()
        self.prefix = prefix
        self.postfix = postfix
        self.data = data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        # make predictions
        x = self.data
        losses = self.model.evaluate(x, batch_size=self.batch_size, all_losses=True)

        # make losses available in logs
        for i, output_name in enumerate(["loss"] + self.model.output_order):
            logs[self.prefix + output_name + self.postfix] = losses[i]
            #print "\n", losses[i], self.prefix + output_name + self.postfix,

# prepare for training
log.info("prepare for training")
train_iter = batch_generator(word_crop, max_len, batch_size, train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, words2id, words2id_size, skipgram_offsets, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size)
train_snapshot = next(batch_generator(word_crop, max_len, min(len(train_rel_ids), snapshot_size), train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, words2id, words2id_size, skipgram_offsets, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size))
valid_snapshot = next(batch_generator(word_crop, max_len, min(len(valid_rel_ids), snapshot_size), valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, words2id, words2id_size, skipgram_offsets, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size))
csv_fields = [
    'experiment', 'epoch',
    'loss', 'val_loss',
    'x_skipgram_loss', 'val_x_skipgram_loss',
    'x_pos_tags_loss', 'val_x_pos_tags_loss',
    'x_rel_marking_loss', 'val_x_rel_marking_loss',
    #'x_rel_types_loss', 'val_x_rel_types_loss', 'rel_types', 'val_rel_types',
    'x_rel_types_one_loss', 'val_x_rel_types_one_loss', 'rel_types_one', 'val_rel_types_one',
    #'x_rel_senses_loss', 'val_x_rel_senses_loss', 'rel_senses', 'val_rel_senses'
    'x_rel_senses_one_loss', 'val_x_rel_senses_one_loss', 'rel_senses_one', 'val_rel_senses_one',
]
png_fields = [
    ['epoch', 'loss', 'x_skipgram_loss', 'x_pos_tags_loss', 'x_rel_marking_loss', 'x_rel_types_loss', 'rel_types', 'x_rel_types_one_loss', 'rel_types_one', 'x_rel_senses_loss', 'rel_senses', 'x_rel_senses_one_loss', 'rel_senses_one'],
    ['epoch', 'val_loss', 'val_x_skipgram_loss', 'val_x_pos_tags_loss', 'val_x_rel_marking_loss', 'val_rel_types', 'val_x_rel_types_loss', 'val_x_rel_types_one_loss', 'val_rel_types_one', 'val_x_rel_senses_loss', 'val_rel_senses', 'val_x_rel_senses_one_loss', 'val_rel_senses_one'],
]
callbacks = [
    EvaluateAllLosses("", "_loss", train_snapshot, batch_size_valid),
    EvaluateAllLosses("val_", "_loss", valid_snapshot, batch_size_valid),
    RelationMetrics("", train_snapshot, batch_size_valid, word_crop, max_len, train_doc_ids, train_words, train_word_metas, train_pos_tags, train_dependencies, train_parsetrees, train_rel_ids, train_rel_parts, train_rel_types, train_rel_senses, train_relations_gold, words2id, words2id_size, skipgram_offsets, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size),
    RelationMetrics("val_", valid_snapshot, batch_size_valid, word_crop, max_len, valid_doc_ids, valid_words, valid_word_metas, valid_pos_tags, valid_dependencies, valid_parsetrees, valid_rel_ids, valid_rel_parts, valid_rel_types, valid_rel_senses, valid_relations_gold, words2id, words2id_size, skipgram_offsets, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size),
    #CSVHistory(metrics_csv, csv_fields, others={"experiment": args.experiment_dir}),
    PlotHistory(metrics_png, png_fields, metrics_csv, csv_fields, others={"experiment": args.experiment_dir}),
    ModelCheckpoint(filepath=weights_hdf5),
    ModelCheckpoint(monitor='loss', mode='min', filepath=weights_hdf5, save_best_only=True),
    EarlyStopping(monitor='loss', mode='min', patience=epochs_patience),
]

# train model
log.info("train model")
model.fit_generator(train_iter, nb_epoch=epochs, samples_per_epoch=len(train_rel_ids), validation_data=valid_snapshot, callbacks=callbacks)
log.info("finished training")

# predict model
# log.info("predict model")
