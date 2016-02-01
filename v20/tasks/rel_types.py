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
from keras.layers.core import Activation, TimeDistributedDense

from common import build_index
from conll16st.relations import tag_to_rtsip


### Model

def rel_types_model(model, ins, max_len, embedding_dim, rel_types2id_size, pre='rtypes'):
    """Discourse relation types model as Keras Graph."""

    # Discourse relation types dense neural network (sample, time_pad, rel_types2id)
    model.add_node(TimeDistributedDense(rel_types2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')
    return pre + '_softmax'


### Build index

def build_rel_types2id(rel_types, max_size=None, min_count=1, rel_types2id=None):
    """Build vocabulary index for all discourse relation types (reserved ids: 0 = padding, 1 = out-of-vocabulary)."""

    return build_index(rel_types, max_size=max_size, min_count=min_count, index=rel_types2id)


### Encode data

def encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_weights, rel_types2id_size, max_len, oov_key=""):
    """Encode discourse relation types as normalized vectors (sample, time_pad, rel_types2id)."""

    # crop sequence if needed
    word_metas_slice = word_metas_slice[:max_len]

    # encode all relation types in same row
    x = np.zeros((max_len, rel_types2id_size), dtype=np.float32)

    for i in range(max_len):
        if i < len(word_metas_slice):
            # mark all relation types
            for rel_tag in word_metas_slice[i]['RelationTags']:
                rel_type, _, _, _ = tag_to_rtsip(rel_tag)

                try:
                    x[i, rel_types2id[rel_type]] += 1.
                except KeyError:  # missing in vocabulary
                    x[i, rel_types2id[oov_key]] += 1.

        if i >= len(word_metas_slice) or not word_metas_slice[i]['RelationTags']:
            # no relation types present
            x[i, rel_types2id[None]] += 1.

    # normalize by rows to [0,1] interval
    x_sum = np.sum(x, axis=1)
    x = (x.T / x_sum).T
    return x


### Tests

def test_build_rel_types2id():
    rel_types = {14903: 'Implicit', 14904: 'Explicit', 14878: 'Explicit'}
    t_rel_types2id = {None: 0, "": 1, "Explicit": 2, "Implicit": 3}
    t_rel_types2id_weights = dict([ (k, 1.) for k in t_rel_types2id ])
    t_rel_types2id_size = len(t_rel_types2id)

    rel_types2id, rel_types2id_weights, rel_types2id_size = build_rel_types2id(rel_types)
    assert rel_types2id == t_rel_types2id
    assert rel_types2id_weights == t_rel_types2id_weights
    assert rel_types2id_size == t_rel_types2id_size

def test_encode_x_rel_types():
    word_metas_slice = [
        {'RelationIDs': [14903], 'SentenceID': 30, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 854, 'SentenceOffset': 827, 'Text': 'trading', 'ParagraphID': 13, 'RelationSpans': ['Arg1']},
        {'RelationIDs': [], 'SentenceID': 30, 'RelationTags': [], 'DocID': 'wsj_1000', 'TokenID': 855, 'SentenceOffset': 827, 'Text': '.', 'ParagraphID': 13, 'RelationSpans': []},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 860, 'SentenceOffset': 857, 'Text': '``', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 861, 'SentenceOffset': 857, 'Text': 'having', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
    ]
    rel_types2id = {None: 0, '': 1, 'Implicit': 2, 'Explicit': 3, 'EntRel': 4}
    rel_types2id_weights = dict([ (k, 1.) for k in rel_types2id ])
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

    x_0 = encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_weights, rel_types2id_size, max_len_0)
    assert (x_0 == t_x_0).all()

    x_1 = encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_weights, rel_types2id_size, max_len_1)
    assert (x_1 == t_x_1).all()

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
