#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Discourse relation types model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Permute

from common import build_index
from conll16st.relations import tag_to_rtsip, filter_tags


### Model

def rel_types_model(model, ins, max_len, embedding_dim, rel_types2id_size, focus, pre='rtypes'):
    """Discourse relation types model as Keras Graph."""

    # prepare focus dimensionality
    model.add_node(RepeatVector(rel_types2id_size), name=pre + '_focus_rep', input=focus)
    model.add_node(Permute((2, 1)), name=pre + '_focus', input=pre + '_focus_rep')

    # discourse relation types dense neural network (sample, time_pad, rel_types2id)
    model.add_node(TimeDistributedDense(rel_types2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')

    # multiplication to focus the activations (doc, time_pad, rel_types2id)
    model.add_node(Activation('linear'), name=pre + '_out', inputs=[pre + '_softmax', pre + '_focus'], merge_mode='mul')
    return pre + '_out'


### Build index

def build_rel_types2id(rel_types, max_size=None, min_count=1, rel_types2id=None):
    """Build vocabulary index for all discourse relation types (reserved ids: 0 = missing, 1 = out-of-vocabulary)."""

    return build_index(rel_types, max_size=max_size, min_count=min_count, index=rel_types2id)


### Encode data

def encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_size, max_len, filter_prefixes=None, oov_key=""):
    """Encode discourse relation types as normalized vectors (sample, time_pad, rel_types2id)."""

    # crop sequence if needed
    word_metas_slice = word_metas_slice[:max_len]

    # encode all relation types in same row
    x = np.zeros((max_len, rel_types2id_size), dtype=np.float32)

    for i in range(max_len):
        if i < len(word_metas_slice):
            # mark all relation types
            tags = filter_tags(word_metas_slice[i]['RelationTags'], filter_prefixes)
            for rel_tag in tags:
                rel_type, _, _, _ = tag_to_rtsip(rel_tag)

                try:
                    x[i, rel_types2id[rel_type]] += 1.
                except KeyError:  # missing in vocabulary
                    x[i, rel_types2id[oov_key]] += 1.

        if i >= len(word_metas_slice) or not filter_tags(word_metas_slice[i]['RelationTags'], filter_prefixes):
            # no relation types present
            x[i, rel_types2id[None]] += 1.

    # normalize by rows to [0,1] interval
    x_sum = np.sum(x, axis=1)
    x2 = (x.T / x_sum).T
    x2[x_sum == 0.] = x[x_sum == 0.]  # prevent NaN
    return x2


def decode_x_rel_types(x_rel_types, token_range, relation, rel_types2id, rel_types2id_size):
    """Decode one discourse relation type for a given relation spans."""

    # sum type predictions for relation tokens
    totals = np.zeros((rel_types2id_size,))
    for i, token_id in enumerate(token_range):
        if token_id in relation['Arg1'] or token_id in relation['Arg2'] or token_id in relation['Connective'] or token_id in relation['Punctuation']:
            totals += x_rel_types[i]
            #XXX: / np.max(x_rel_types[i])

    # return most probable type
    rel_type = None
    max_total = -1.
    for t, j in rel_types2id.items():
        if totals[j] > max_total:
            max_total = totals[j]
            rel_type = t
    return rel_type, totals


### Tests

def test_build_rel_types2id():
    rel_types = {14903: 'Implicit', 14904: 'Explicit', 14878: 'Explicit'}
    t_rel_types2id = {None: 0, "": 1, "Explicit": 2, "Implicit": 3}
    t_rel_types2id_size = len(t_rel_types2id)

    rel_types2id, rel_types2id_size = build_rel_types2id(rel_types)
    assert rel_types2id == t_rel_types2id
    assert rel_types2id_size == t_rel_types2id_size

def test_encode_x_rel_types():
    word_metas_slice = [
        {'RelationIDs': [14903], 'SentenceID': 30, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 854, 'SentenceOffset': 827, 'Text': 'trading', 'ParagraphID': 13, 'RelationSpans': ['Arg1']},
        {'RelationIDs': [], 'SentenceID': 30, 'RelationTags': [], 'DocID': 'wsj_1000', 'TokenID': 855, 'SentenceOffset': 827, 'Text': '.', 'ParagraphID': 13, 'RelationSpans': []},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 859, 'SentenceOffset': 857, 'Text': 'that', 'ParagraphID': 13, 'RelationParts': ['Arg2', 'Arg1']},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 860, 'SentenceOffset': 857, 'Text': '``', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
    ]
    rel_types2id = {None: 0, '': 1, 'Implicit': 2, 'Explicit': 3, 'EntRel': 4}
    rel_types2id_size = len(rel_types2id)
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

    x_0 = encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_size, max_len_0)
    assert (x_0 == t_x_0).all()

    x_1 = encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_size, max_len_1)
    assert (x_1 == t_x_1).all()

def test_decode_x_rel_types():
    rel_types2id = {None: 0, '': 1, 'Implicit': 2, 'Explicit': 3, 'EntRel': 4}
    rel_types2id_size = len(rel_types2id)
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
    x_rel_types_0 = [
        [0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0., 0.5, 0.5, 0.],
        [0., 0., 0.5, 0.5, 0.],
    ]
    t_type_0 = 'Implicit'
    x_rel_types_1 = [
        [0., 0.25, 0., 0.75, 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0.5, 0., 0.5, 0.],
        [0., 0., 0., 0.5, 0.5],
    ]
    t_type_1 = 'Explicit'

    type_0 = decode_x_rel_types(x_rel_types_0, range(token_start, token_end), relation, rel_types2id, rel_types2id_size)
    assert type_0 == t_type_0

    type_1 = decode_x_rel_types(x_rel_types_1, range(token_start, token_end), relation, rel_types2id, rel_types2id_size)
    assert type_1 == t_type_1

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
