#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Discourse relation senses model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Permute
from keras.layers.recurrent import GRU

from common import build_index
from conll16st.relations import tag_to_rtsip, filter_tags


### Model

def rel_senses_model(model, ins, max_len, embedding_dim, rel_senses2id_size, focus, pre='rsenses'):
    """Discourse relation senses model as Keras Graph."""

    # prepare focus dimensionality (sample, time_pad, rel_senses2id)
    model.add_node(RepeatVector(rel_senses2id_size), name=pre + '_focus_rep', input=focus)
    model.add_node(Permute((2, 1)), name=pre + '_focus', input=pre + '_focus_rep')

    # discourse relation senses dense neural network (sample, time_pad, rel_senses2id)
    model.add_node(TimeDistributedDense(rel_senses2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')

    # multiplication to focus the activations (sample, time_pad, rel_senses2id)
    model.add_node(Activation('linear'), name=pre + '_out', inputs=[pre + '_focus', pre + '_softmax'], merge_mode='mul')
    return pre + '_out'


def rel_senses_one_model(model, ins, max_len, embedding_dim, rel_senses2id_size, focus, pre='rsensesone'):
    """Discourse relation senses model to return one output as Keras Graph."""

    # forward recurrent layer returning one last output (sample, rel_senses2id)
    model.add_node(GRU(rel_senses2id_size, return_sequences=False, activation='sigmoid', inner_activation='sigmoid', init='he_uniform', inner_init='orthogonal'), name=pre + '_fwd', input=ins[0])

    # backward recurrent layer returning one last output (sample, rel_senses2id)
    model.add_node(GRU(rel_senses2id_size, return_sequences=False, activation='sigmoid', inner_activation='sigmoid', init='he_uniform', inner_init='orthogonal', go_backwards=True), name=pre + '_bck', input=ins[0])

    # join activations (sample, rel_senses2id)
    model.add_node(Activation('linear'), name=pre + '_out', inputs=[pre + '_fwd', pre + '_bck'], merge_mode='ave')
    return pre + '_out'


### Build index

def build_rel_senses2id(rel_senses, max_size=None, min_count=1, rel_senses2id=None):
    """Build vocabulary index for all discourse relation senses (reserved ids: 0 = missing, 1 = out-of-vocabulary)."""

    return build_index(rel_senses, max_size=max_size, min_count=min_count, index=rel_senses2id)


### Encode data

def encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_size, max_len, filter_prefixes=None, oov_key=""):
    """Encode discourse relation senses as normalized vectors (sample, time_pad, rel_senses2id)."""

    # crop sequence if needed
    word_metas_slice = word_metas_slice[:max_len]

    # encode all relation senses in same row
    x = np.zeros((max_len, rel_senses2id_size), dtype=np.float32)

    for i in range(max_len):
        if i < len(word_metas_slice):
            # mark all relation senses
            tags = filter_tags(word_metas_slice[i]['RelationTags'], filter_prefixes)
            for rel_tag in tags:
                _, rel_sense, _, _ = tag_to_rtsip(rel_tag)

                try:
                    x[i, rel_senses2id[rel_sense]] += 1.
                except KeyError:  # missing in vocabulary
                    x[i, rel_senses2id[oov_key]] += 1.

        if i >= len(word_metas_slice) or not filter_tags(word_metas_slice[i]['RelationTags'], filter_prefixes):
            # no relation senses present
            x[i, rel_senses2id[None]] += 1.

    # normalize by rows to [0,1] interval
    x_sum = np.sum(x, axis=1)
    x2 = (x.T / x_sum).T
    x2[x_sum == 0.] = x[x_sum == 0.]  # prevent NaN
    return x2


def decode_x_rel_senses(x_rel_senses, token_range, relation, rel_senses2id, rel_senses2id_size):
    """Decode one discourse relation sense for a given relation spans."""

    # sum sense predictions for relation tokens
    totals = np.zeros((rel_senses2id_size,))
    for i, token_id in enumerate(token_range):
        if token_id in relation['Arg1'] or token_id in relation['Arg2'] or token_id in relation['Connective'] or token_id in relation['Punctuation']:
            totals += x_rel_senses[i]
            #XXX: / np.max(x_rel_senses[i])

    # return most probable sense
    rel_sense = None
    max_total = -1.
    for t, j in rel_senses2id.items():
        if totals[j] > max_total:
            max_total = totals[j]
            rel_sense = t
    return rel_sense, totals


def encode_x_rel_senses_one(rel_sense, rel_senses2id, rel_senses2id_size, oov_key=""):
    """Encode discourse relation senses as one normalized vector (sample, rel_senses2id)."""

    # one-hot encode sense
    try:
        i = rel_senses2id[rel_sense]
    except KeyError:  # missing in vocabulary
        i = rel_senses2id[oov_key]
    x = np.zeros((rel_senses2id_size,), dtype=np.float32)
    x[i] = 1.
    return x


def decode_x_rel_senses_one(x_rel_senses_one, rel_senses2id, rel_senses2id_size):
    """Decode one discourse relation sense from a normalized vector."""

    # normalize by rows to [0,1] interval
    x_sum = np.sum(x_rel_senses_one)
    totals = x_rel_senses_one / x_sum
    totals[x_sum == 0.] = x_rel_senses_one[x_sum == 0.]  # prevent NaN

    # return most probable sense
    rel_sense = None
    max_total = -1.
    for t, j in rel_senses2id.items():
        if totals[j] > max_total:
            max_total = totals[j]
            rel_sense = t
    return rel_sense, totals


### Tests

def test_build_rel_senses2id():
    rel_senses = {14903: 'Comparison.Contrast', 14904: 'Comparison.Concession', 14878: 'Comparison.Contrast'}
    t_rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3}
    t_rel_senses2id_size = len(t_rel_senses2id)

    rel_senses2id, rel_senses2id_size = build_rel_senses2id(rel_senses)
    assert rel_senses2id == t_rel_senses2id
    assert rel_senses2id_size == t_rel_senses2id_size

def test_encode_x_rel_senses():
    word_metas_slice = [
        {'RelationIDs': [14903], 'SentenceID': 30, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 854, 'SentenceOffset': 827, 'Text': 'trading', 'ParagraphID': 13, 'RelationSpans': ['Arg1']},
        {'RelationIDs': [], 'SentenceID': 30, 'RelationTags': [], 'DocID': 'wsj_1000', 'TokenID': 855, 'SentenceOffset': 827, 'Text': '.', 'ParagraphID': 13, 'RelationSpans': []},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 860, 'SentenceOffset': 857, 'Text': '``', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 861, 'SentenceOffset': 857, 'Text': 'having', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
    ]
    rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3, "Contingency.Condition": 4}
    rel_senses2id_size = len(rel_senses2id)
    max_len_0 = 3
    max_len_1 = 5
    t_x_0 = [
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0],
    ]
    t_x_1 = [
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0],
        [0, 0, 0.5, 0.5, 0],
        [1, 0, 0, 0, 0],
    ]

    x_0 = encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_size, max_len_0)
    assert (x_0 == t_x_0).all()

    x_1 = encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_size, max_len_1)
    assert (x_1 == t_x_1).all()

def test_decode_x_rel_types():
    rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3, "Contingency.Condition": 4}
    rel_senses2id_size = len(rel_senses2id)
    relation = {
        'Arg1': [854],
        'Arg1Len': 1,
        'Arg2': [859, 860],
        'Arg2Len': 2,
        'Connective': [],
        'ConnectiveLen': 0,
        'Punctuation': [],
        'PunctuationLen': 0,
        'PunctuationType': '',
        'DocID': 'wsj_1000',
        'ID': 14903,
        'TokenMin': 854,
        'TokenMax': 861,
        'TokenCount': 3,
    }
    token_start = 854
    token_end = 861
    x_rel_senses_0 = [
        [0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0., 0.5, 0.5, 0.],
        [0., 0., 0.5, 0.5, 0.],
    ]
    t_sense_0 = 'Comparison.Contrast'
    x_rel_senses_1 = [
        [0., 0.25, 0., 0.75, 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0.5, 0., 0.5, 0.],
        [0., 0., 0., 0.5, 0.5],
    ]
    t_sense_1 = 'Comparison.Concession'

    sense_0, totals_0 = decode_x_rel_senses(x_rel_senses_0, range(token_start, token_end), relation, rel_senses2id, rel_senses2id_size)
    assert sense_0 == t_sense_0

    sense_1, totals_1 = decode_x_rel_senses(x_rel_senses_1, range(token_start, token_end), relation, rel_senses2id, rel_senses2id_size)
    assert sense_1 == t_sense_1

def test_encode_x_rel_senses_one():
    rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3, "Contingency.Condition": 4}
    rel_senses2id_size = len(rel_senses2id)
    rel_sense_0 = "Comparison.Contrast"
    t_x_0 = [0, 0, 1, 0, 0]
    rel_sense_1 = "Expansion.Alternative"
    t_x_1 = [0, 1, 0, 0, 0]

    x_0 = encode_x_rel_senses_one(rel_sense_0, rel_senses2id, rel_senses2id_size)
    assert (x_0 == t_x_0).all()

    x_1 = encode_x_rel_senses_one(rel_sense_1, rel_senses2id, rel_senses2id_size)
    assert (x_1 == t_x_1).all()

def test_decode_x_rel_types():
    rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3, "Contingency.Condition": 4}
    rel_senses2id_size = len(rel_senses2id)
    x_rel_senses_one_0 = [0, 0, 1, 0, 0]
    t_sense_0 = "Comparison.Contrast"
    x_rel_senses_one_1 = [0, 1, 0, 0, 0]
    t_sense_1 = ""

    sense_0, totals_0 = decode_x_rel_senses_one(x_rel_senses_one_0, rel_senses2id, rel_senses2id_size)
    assert sense_0 == t_sense_0

    sense_1, totals_1 = decode_x_rel_senses_one(x_rel_senses_one_1, rel_senses2id, rel_senses2id_size)
    assert sense_1 == t_sense_1

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
