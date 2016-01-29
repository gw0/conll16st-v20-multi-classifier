#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Discourse relation senses model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

from common import build_index
from conll16st.relations import tag_to_rtsip


### Model


### Build index

def build_rel_senses2id(rel_senses, max_size=None, min_count=1, pos_tags2id=None):
    """Build vocabulary index for all discourse relation senses (reserved ids: 0 = padding, 1 = out-of-vocabulary)."""

    return build_index(rel_senses, max_size=max_size, min_count=min_count, index=pos_tags2id)


### Encode

def encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_weights, rel_senses2id_size, max_len, oov_key=""):
    """Encode discourse relation senses as normalized vectors (sample, time_pad, rel_senses2id)."""

    # crop sequence if needed
    word_metas_slice = word_metas_slice[:max_len]

    # encode all relation senses in same row
    x = np.zeros((max_len, rel_senses2id_size), dtype=np.float32)

    for i in range(max_len):
        if i < len(word_metas_slice):
            # mark all relation senses
            for rel_tag in word_metas_slice[i]['RelationTags']:
                _, rel_sense, _, _ = tag_to_rtsip(rel_tag)

                try:
                    x[i, rel_senses2id[rel_sense]] += 1.
                except KeyError:  # missing in vocabulary
                    x[i, rel_senses2id[oov_key]] += 1.

        if i >= len(word_metas_slice) or not word_metas_slice[i]['RelationTags']:
            # no relation senses present
            x[i, rel_senses2id[None]] += 1.

    # normalize by rows to [0,1] interval
    x_sum = np.sum(x, axis=1)
    x = (x.T / x_sum).T
    return x


### Tests

def test_build_rel_senses2id():
    rel_senses = {14903: 'Comparison.Contrast', 14904: 'Comparison.Concession', 14878: 'Comparison.Contrast'}
    t_rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3}
    t_rel_senses2id_weights = dict([ (k, 1.) for k in t_rel_senses2id ])
    t_rel_senses2id_size = len(t_rel_senses2id)

    rel_senses2id, rel_senses2id_weights, rel_senses2id_size = build_rel_senses2id(rel_senses)
    assert rel_senses2id == t_rel_senses2id
    assert rel_senses2id_weights == t_rel_senses2id_weights
    assert rel_senses2id_size == t_rel_senses2id_size

def test_encode_x_rel_senses():
    word_metas_slice = [
        {'RelationIDs': [14903], 'SentenceID': 30, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 854, 'SentenceOffset': 827, 'Text': 'trading', 'ParagraphID': 13, 'RelationSpans': ['Arg1']},
        {'RelationIDs': [], 'SentenceID': 30, 'RelationTags': [], 'DocID': 'wsj_1000', 'TokenID': 855, 'SentenceOffset': 827, 'Text': '.', 'ParagraphID': 13, 'RelationSpans': []},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 860, 'SentenceOffset': 857, 'Text': '``', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
        {'RelationIDs': [14903, 14904], 'SentenceID': 31, 'RelationTags': ['Implicit:Comparison.Contrast:14903:Arg2', 'Explicit:Comparison.Concession:14904:Arg1'], 'DocID': 'wsj_1000', 'TokenID': 861, 'SentenceOffset': 857, 'Text': 'having', 'ParagraphID': 13, 'RelationSpans': ['Arg2', 'Arg1']},
    ]
    rel_senses2id = {None: 0, "": 1, "Comparison.Contrast": 2, "Comparison.Concession": 3, "Contingency.Condition": 4}
    rel_senses2id_weights = dict([ (k, 1.) for k in rel_senses2id ])
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

    x_0 = encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_weights, rel_senses2id_size, max_len_0)
    assert (x_0 == t_x_0).all()

    x_1 = encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_weights, rel_senses2id_size, max_len_1)
    assert (x_1 == t_x_1).all()

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
