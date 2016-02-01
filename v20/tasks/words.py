#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Word embedding task.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

from common import build_index, map_sequence, pad_sequence


### Build index

def build_words2id(words, max_size=None, min_count=1, words2id=None):
    """Build vocabulary index for all words (reserved ids: 0 = padding, 1 = out-of-vocabulary)."""

    return build_index(words, max_size=max_size, min_count=min_count, index=words2id)


### Encode data

def encode_x_words(words_slice, words2id, words2id_weights, words2id_size, max_len):
    """Encode words as word ids with masked post-padding (sample, time_pad)."""

    ids = map_sequence(words_slice, words2id)
    x_words = pad_sequence(ids, max_len)
    return x_words


def encode_x_words_rand(words_slice, words2id, words2id_weights, words2id_size, max_len):
    """Encode words as word ids with random post-padding (sample, time_pad)."""

    ids = map_sequence(words_slice, words2id)
    x_words_rand = pad_sequence(ids, max_len, value='rand', max_rand=words2id_size)
    return x_words_rand


### Tests

def test_build_words():
    words = {}
    words["wsj_1000"] = ["Kemper", "Financial", "Services", "Inc.", ",", "charging", "Kemper"]
    words["wsj_1000x"] = words["wsj_1000"][:3]
    min_count = 2
    t_words2id = {None: 0, "": 1, "Kemper": 2, "Financial": 3, "Services": 4}
    t_words2id_weights = dict([ (k, 1.) for k in t_words2id ])
    t_words2id_size = len(t_words2id)

    words2id, words2id_weights, words2id_size = build_words2id(words, min_count=min_count)
    assert words2id == t_words2id
    assert words2id_weights == t_words2id_weights
    assert words2id_size == t_words2id_size

def test_encode_x_words():
    words_slice = ["Kemper", "Financial", "Services", "Inc.", ",", "charging", "Kemper"]
    words2id = {None: 0, "": 1, "Kemper": 2, "Financial": 3, "Services": 4, "Inc.": 5, ",": 6, "charging": 7}
    words2id_weights = dict([ (k, 1.) for k in words2id ])
    words2id_size = len(words2id)
    max_len_0 = 3
    max_len_1 = 10
    t_x_0 = [2, 3, 4]
    t_x_1 = [2, 3, 4, 5, 6, 7, 2, 0, 0, 0]

    x_0 = encode_x_words(words_slice, words2id, words2id_weights, words2id_size, max_len_0)
    assert (x_0 == t_x_0).all()

    x_0_rand = encode_x_words_rand(words_slice, words2id, words2id_weights, words2id_size, max_len_0)
    assert (x_0_rand == t_x_0).all()

    x_1 = encode_x_words(words_slice, words2id, words2id_weights, words2id_size, max_len_1)
    assert (x_1 == t_x_1).all()

    x_1_rand = encode_x_words_rand(words_slice, words2id, words2id_weights, words2id_size, max_len_1)
    assert (x_1_rand[:len(words_slice)] == t_x_1[:len(words_slice)]).all()
    assert (x_1_rand[len(words_slice):] != t_x_1[len(words_slice):]).all()

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
