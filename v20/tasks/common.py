#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Common helper functions.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import cPickle as pickle
import numpy as np


def conv_window_to_offsets(window_size, negative_samples, word_crop):
    """Convert window size and negative samples to list of offsets."""

    offsets = range(-(window_size - 1) // 2, window_size // 2 + 1)
    if window_size % 2 == 0:
        del offsets[window_size // 2]
    for i in range(negative_samples):
        offsets.append(word_crop + i)
    return offsets


def build_index(sequences, max_size=None, min_count=1, index=None):
    """Build vocabulary index from dicts or lists of strings (reserved ids: 0 = padding, 1 = out-of-vocabulary)."""
    if index is None:
        index = {}

    def _traverse_cnt(obj, cnts):
        """Recursively traverse dicts and lists of strings."""
        if isinstance(obj, dict):
            for s in obj.itervalues():
                _traverse_cnt(s, cnts)
        elif isinstance(obj, list):
            for s in obj:
                _traverse_cnt(s, cnts)
        else:
            try:
                cnts[obj] += 1
            except KeyError:
                cnts[obj] = 1

    # count string occurrences
    cnts = {}
    _traverse_cnt(sequences, cnts)

    # ignore strings with low occurrences
    for k, cnt in cnts.items():
        if cnt < min_count:
            del cnts[k]

    # rank strings by decreasing occurrences and use as index
    index_rev = [None, ""] + sorted(cnts, key=cnts.get, reverse=True)
    if max_size is not None:
        index_rev = index_rev[:max_size]

    # mapping of strings to vocabulary ids
    index.update([ (k, i) for i, k in enumerate(index_rev) ])

    # largest vocabulary id + 1
    index_size = len(index)
    return index, index_size


def map_sequence(sequence, index, oov_key=""):
    """Map sequence of strings to vocabulary ids."""

    ids = []
    for s in sequence:
        try:
            ids.append(index[s])
        except KeyError:  # missing in vocabulary
            ids.append(index[oov_key])
    return ids


def pad_sequence(sequence, max_len, value=0, max_rand=None):
    """Post-pad sequence of ids as numpy array."""

    # crop sequence if needed
    sequence = sequence[:max_len]

    # convert to numpy array with masked and random post-padding
    if isinstance(value, int):
        x = np.hstack([sequence, np.ones((max_len - len(sequence),), dtype=np.int) * value])
    elif isinstance(value, float):
        x = np.hstack([sequence, np.ones((max_len - len(sequence),), dtype=np.float32) * value])
    elif value == 'rand' and isinstance(max_rand, int):
        x = np.hstack([sequence, np.random.randint(1, max_rand, size=max_len - len(sequence),)])
    else:
        raise ValueError("Padding value '{}' not understood".format(value))
    return x


def onehot_sequence(sequence, max_len, index_size, value_zero=0, value_one=1):
    """Encode sequence of ids to one-hot vectors as numpy arrays."""

    # crop sequence if needed
    sequence = sequence[:max_len]

    # map ids to one-hot encoding
    x = np.ones((max_len, index_size), dtype=np.int) * value_zero
    x[np.arange(max_len), np.hstack([sequence, np.zeros((max_len - len(sequence),), dtype=np.int)])] = value_one
    return x


def save_to_pkl(pkl, obj):
    """Save experiment resource, such as vocabulary index."""

    with open(pkl, 'wb') as f:
        pickle.dump(obj, f)
    return obj


def load_from_pkl(pkl):
    """Load experiment resource, such as vocabulary index."""

    with open(pkl, 'rb') as f:
        return pickle.load(f)
    return None


### Tests

def test_window_to_offsets():
    window_size = 4
    negative_samples = 0
    word_crop = 100
    t_offsets = [-2, -1, 1, 2]

    offsets = conv_window_to_offsets(window_size, negative_samples, word_crop)
    assert offsets == t_offsets

def test_window_to_offsets_neg():
    window_size = 5
    negative_samples = window_size
    word_crop = 100
    t_offsets = [-2, -1, 0, 1, 2, 100, 101, 102, 103, 104]

    offsets = conv_window_to_offsets(window_size, negative_samples, word_crop)
    assert offsets == t_offsets

def test_index_dict():
    sequences = {1: 'foo', 2: 'foo', 10: 'bar'}
    t_index = {None: 0, "": 1, "foo": 2, "bar": 3}
    t_index_size = len(t_index)

    index, index_size = build_index(sequences)
    assert index == t_index
    assert index_size == t_index_size

def test_index_list_of_lists():
    sequences = [
        ["a", "b", "c", "a"],
        ["a", "b", "d"],
    ]
    min_count = 2
    t_index = {None: 0, "": 1, "a": 2, "b": 3}
    t_index_size = len(t_index)

    index, index_size = build_index(sequences, min_count=min_count)
    assert index == t_index
    assert index_size == t_index_size

def test_map_sequence():
    sequence = ["a", "b", "c", "a"]
    index = {None: 0, "": 1, "a": 2, "b": 3, "c": 4, "d": 5}
    t_ids = [2, 3, 4, 2]

    ids = map_sequence(sequence, index)
    assert ids == t_ids

def test_pad_sequence():
    ids = [2, 3, 4, 2]
    index_size = 6
    max_len_0 = 3
    max_len_1 = 5
    max_len_2 = max_len_1
    value_2 = 'rand'
    t_x_0 = [2, 3, 4]
    t_x_1 = [2, 3, 4, 2, 0]
    t_x_2 = t_x_1

    x_0 = pad_sequence(ids, max_len_0, value=0)
    assert (x_0 == t_x_0).all()

    x_1 = pad_sequence(ids, max_len_1, value=0)
    assert (x_1 == t_x_1).all()

    x_2 = pad_sequence(ids, max_len_2, value=value_2, max_rand=index_size)
    assert (x_2[:4] == t_x_2[:4]).all()
    assert (x_2[4:] != t_x_2[4:]).all()

def test_onehot_sequence():
    ids = [2, 3, 4, 2]
    index_size = 6
    max_len_0 = 3
    max_len_1 = 5
    t_onehots_0 = [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ]
    t_onehots_1 = [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]

    onehots_0 = onehot_sequence(ids, max_len_0, index_size)
    assert (onehots_0 == t_onehots_0).all()

    onehots_1 = onehot_sequence(ids, max_len_1, index_size)
    assert (onehots_1 == t_onehots_1).all()

def test_save_load_pkl(tmpdir):
    pkl = str(tmpdir.join("test_save_load_pkl.pkl"))
    t_index = {None: 0, "": 1, "foo": 2, "bar": 3}
    t_index_size = len(t_index)
    t_obj = (t_index, t_index_size)

    obj = save_to_pkl(pkl, t_obj)
    assert obj == t_obj

    obj = load_from_pkl(pkl)
    assert obj == t_obj

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
