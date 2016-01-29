#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
POS tagging model/task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

from keras.layers.core import Activation, TimeDistributedDense

from common import build_index, map_sequence, onehot_sequence, pad_sequence


### Model

def pos_tags_model(model, ins, max_len, embedding_dim, pos_tags2id_size, pre='pos'):
    """POS tagging model as Keras Graph."""

    # POS tag dense neural network (doc, time_pad, pos_tags2id)
    model.add_node(TimeDistributedDense(pos_tags2id_size, init='he_uniform'), name=pre + '_dense', input=ins[0])
    model.add_node(Activation('softmax'), name=pre + '_softmax', input=pre + '_dense')
    return pre + '_softmax'


### Build index

def build_pos_tags2id(pos_tags, max_size=None, min_count=1, pos_tags2id=None):
    """Build vocabulary index for all POS tags (reserved ids: 0 = padding, 1 = out-of-vocabulary)."""

    return build_index(pos_tags, max_size=max_size, min_count=min_count, index=pos_tags2id)


### Encode

def encode_x_pos_tags(pos_tags_slice, pos_tags2id, pos_tags2id_weights, pos_tags2id_size, max_len):
    """Encode POS tags as one-hot vectors (sample, time_pad, pos_tags2id)."""

    ids = map_sequence(pos_tags_slice, pos_tags2id)
    x_pos_tags = onehot_sequence(ids, max_len, pos_tags2id_size)
    return x_pos_tags


### Tests

def test_build_pos_tags():
    pos_tags = {}
    pos_tags["wsj_1000"] = ["NNP", "NNP", "NNPS", "NNP", ",", "VBG", "NNP"]
    pos_tags["wsj_1000x"] = pos_tags["wsj_1000"][:3]
    min_count = 2
    t_pos_tags2id = {None: 0, "": 1, "NNP": 2, "NNPS": 3}
    t_pos_tags2id_weights = dict([ (k, 1.) for k in t_pos_tags2id ])
    t_pos_tags2id_size = len(t_pos_tags2id)

    pos_tags2id, pos_tags2id_weights, pos_tags2id_size = build_pos_tags2id(pos_tags, min_count=min_count)
    assert pos_tags2id == t_pos_tags2id
    assert pos_tags2id_weights == t_pos_tags2id_weights
    assert pos_tags2id_size == t_pos_tags2id_size

def test_encode_x_pos_tags():
    pos_tags_slice = ["NNP", "NNP", "NNPS", "NNP", ",", "VBG", "NNP"]
    pos_tags2id = {None: 0, "": 1, "NNP": 2, "NNPS": 3, ",": 4, "VBG": 5}
    pos_tags2id_weights = dict([ (k, 1.) for k in pos_tags2id ])
    pos_tags2id_size = len(pos_tags2id)
    max_len_0 = 3
    max_len_1 = 10
    t_x_0 = [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
    t_x_1 = [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]

    x_0 = encode_x_pos_tags(pos_tags_slice, pos_tags2id, pos_tags2id_weights, pos_tags2id_size, max_len_0)
    assert (x_0 == t_x_0).all()

    x_1 = encode_x_pos_tags(pos_tags_slice, pos_tags2id, pos_tags2id_weights, pos_tags2id_size, max_len_1)
    assert (x_1 == t_x_1).all()

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
