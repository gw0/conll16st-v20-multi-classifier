#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Discourse relation marking model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from keras.layers.core import Activation, TimeDistributedDense

from conll16st.relations import tag_to_rtsip, filter_tags


### Model

def rel_marking_model(model, ins, max_len, embedding_dim, rel_marking2id_size, pre='rmarking'):
    """Discourse relation marking model as Keras Graph."""

    # Discourse relation marking dense neural network (sample, time_pad, rel_marking2id)
    model.add_node(TimeDistributedDense(rel_marking2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')
    return pre + '_softmax'


### Build index

def build_rel_marking2id(mode='IO'):
    """Build vocabulary index for all discourse relation boundary marking (reserved ids: 0 = other)."""

    if mode == 'IOBES-part':
        rel_marking2id = {
            "O": [0],
            "B-Arg1": [1],
            "I-Arg1": [2],
            "E-Arg1": [3],
            "S-Arg1": [4],
            "B-Arg2": [5],
            "I-Arg2": [6],
            "E-Arg2": [7],
            "S-Arg2": [8],
            "B-Connective": [9],
            "I-Connective": [10],
            "E-Connective": [11],
            "S-Connective": [12],
            "B-Punctuation": [13],
            "I-Punctuation": [14],
            "E-Punctuation": [15],
            "S-Punctuation": [16],
        }
    elif mode == 'IO-part':
        rel_marking2id = {
            "O": [0],
            "B-Arg1": [1],
            "I-Arg1": [1],
            "E-Arg1": [1],
            "S-Arg1": [1],
            "B-Arg2": [2],
            "I-Arg2": [2],
            "E-Arg2": [2],
            "S-Arg2": [2],
            "B-Connective": [3],
            "I-Connective": [3],
            "E-Connective": [3],
            "S-Connective": [3],
            "B-Punctuation": [4],
            "I-Punctuation": [4],
            "E-Punctuation": [4],
            "S-Punctuation": [4],
        }
    elif mode == 'I-join':
        rel_marking2id = {
            "O": [],
            "B-Arg1": [0],
            "I-Arg1": [0],
            "E-Arg1": [0],
            "S-Arg1": [0],
            "B-Arg2": [0],
            "I-Arg2": [0],
            "E-Arg2": [0],
            "S-Arg2": [0],
            "B-Connective": [0],
            "I-Connective": [0],
            "E-Connective": [0],
            "S-Connective": [0],
            "B-Punctuation": [0],
            "I-Punctuation": [0],
            "E-Punctuation": [0],
            "S-Punctuation": [0],
        }
    else:  # invalid mode
        return None, None
    rel_marking2id_size = max([ i  for l in rel_marking2id.values() for i in l ]) + 1
    return rel_marking2id, rel_marking2id_size


### Encode data

def encode_x1_rel_marking(word_metas_slice, rel_marking2id, rel_marking2id_size, max_len, filter_prefixes=None):
    """Encode discourse relation boundary markers (sample, time_pad, rel_marking2id)."""

    # crop sequence if needed
    word_metas_slice = word_metas_slice[:max_len]

    # encode all relation boundaries and spans
    x = np.zeros((max_len, rel_marking2id_size), dtype=np.float32)
    last_tags = set([])
    for w1_i in range(max_len):  # iterate word 1

        # filtered word 1 tags by specified relation tags
        w1_tags = []
        if w1_i < len(word_metas_slice):
            w1_tags = word_metas_slice[w1_i]['RelationTags']
            w1_tags = filter_tags(w1_tags, filter_prefixes)

        # filtered word 2 (next word) tags by specified relation tags
        w2_tags = []
        w2_i = w1_i + 1
        if w2_i < len(word_metas_slice):
            w2_tags = word_metas_slice[w2_i]['RelationTags']
            w2_tags = filter_tags(w2_tags, filter_prefixes)

        # mark
        for rel_tag in w1_tags:
            rel_type, rel_sense, rel_id, rel_part = tag_to_rtsip(rel_tag)

            if rel_tag not in last_tags and rel_tag not in w2_tags:
                # mark single tag
                x[w1_i, rel_marking2id["S-{}".format(rel_part)]] += 1
            elif rel_tag not in last_tags:
                # mark begin tag
                x[w1_i, rel_marking2id["B-{}".format(rel_part)]] += 1
            elif rel_tag not in w2_tags:
                # mark end tag
                x[w1_i, rel_marking2id["E-{}".format(rel_part)]] += 1
            else:
                # mark inside tag
                x[w1_i, rel_marking2id["I-{}".format(rel_part)]] += 1

        # mark other tag
        if not np.any(x[w1_i]):
            x[w1_i, rel_marking2id["O"]] += 1

        last_tags = set(w1_tags)

    # normalize by rows to [0,1] interval
    x_sum = np.sum(x, axis=1)
    x2 = (x.T / x_sum).T
    x2[x_sum == 0.] = x[x_sum == 0.]  # prevent NaN
    return x2


def encode_x1_rel_focus(word_metas_slice, max_len, filter_prefixes=None):
    """Encode discourse relation focus markers (sample, time_pad)."""

    rel_marking2id, rel_marking2id_size = build_rel_marking2id(mode='I-join')
    x = encode_x1_rel_marking(word_metas_slice, rel_marking2id, rel_marking2id_size, max_len, filter_prefixes)
    x = x.reshape(x.shape[:-1])  # flatten last dimension
    return x


### Tests

def test_build_rel_marking2id():
    mode_0 = 'IO'
    t_size_0 = 5
    mode_1 = 'IOBES'
    t_size_1 = 17

    rel_marking2id, rel_marking2id_size = build_rel_marking2id(mode_0)
    assert rel_marking2id_size == t_size_0

    rel_marking2id, rel_marking2id_size = build_rel_marking2id(mode_1)
    assert rel_marking2id_size == t_size_1

def test_encode_x1_rel_marking():
    word_metas_slice = [
        {'RelationIDs': [14903], 'SentenceID': 30, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 854, 'SentenceOffset': 827, 'Text': 'trading', 'ParagraphID': 13, 'RelationSpans': ['Arg1']},
        {'RelationIDs': [], 'SentenceID': 30, 'RelationTags': [], 'DocID': 'wsj_1000', 'TokenID': 855, 'SentenceOffset': 827, 'Text': '.', 'ParagraphID': 13, 'RelationSpans': []},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 860, 'SentenceOffset': 857, 'Text': '``', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 861, 'SentenceOffset': 857, 'Text': 'having', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
    ]
    tag = "Implicit:Comparison.Contrast:14903:"
    mode_01 = 'IO'
    max_len_0 = 3
    max_len_1 = 5
    t_x_0 = [
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    t_x_1 = [
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ]
    mode_23 = 'I-join'
    max_len_2 = 3
    max_len_3 = 5
    t_x_2 = [1, 0, 1]
    t_x_3 = [1, 0, 1, 1, 0]

    rel_marking2id, rel_marking2id_size = build_rel_marking2id(mode_01)

    x_0 = encode_x1_rel_marking(word_metas_slice, rel_marking2id, rel_marking2id_size, max_len_0, filter_prefixes=[tag])
    assert (x_0 == t_x_0).all()

    x_1 = encode_x1_rel_marking(word_metas_slice, rel_marking2id, rel_marking2id_size, max_len_1, filter_prefixes=[tag])
    assert (x_1 == t_x_1).all()

    rel_marking2id, rel_marking2id_size = build_rel_marking2id(mode_23)

    x_2 = encode_x1_rel_focus(word_metas_slice, max_len_2, filter_prefixes=[tag])
    assert (x_2 == t_x_2).all()

    x_3 = encode_x1_rel_focus(word_metas_slice, max_len_3, filter_prefixes=[tag])
    assert (x_3 == t_x_3).all()

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
