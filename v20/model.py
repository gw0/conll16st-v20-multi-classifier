#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Deep learning architecture model for discourse relation sense classification (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import random
import numpy as np
from keras.models import make_batches
from keras.models import Graph
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import TimeDistributedDense, Reshape

from tasks.words import encode_x_words, encode_x_words_rand
from tasks.pos_tags import pos_tags_model, encode_x_pos_tags
from tasks.rel_types import rel_types_model, encode_x_rel_types
from tasks.rel_senses import rel_senses_model, encode_x_rel_senses
from tasks.rel_marking import rel_marking_model, encode_x1_rel_marking, encode_x1_rel_focus
from conll16st.relations import rtsip_to_tag


### Model

def build_model(max_len, embedding_dim, words2id_size, pos_tags2id_size, rel_types2id_size, rel_senses2id_size, rel_marking2id_size):

    model = Graph()
    shared_layers = 2
    loss = {
        #'x_skipgram': 'mse',
        #'x_pos_tags': 'binary_crossentropy',
        'x_rel_types': 'binary_crossentropy',
        'x_rel_senses': 'binary_crossentropy',
    }

    # input: word ids with masked post-padding (doc, time_pad)
    model.add_input(name='x_words_pad', input_shape=(None,), dtype='int')

    # input: word ids with random post-padding (doc, time_pad)
    if 'x_skipgram' in loss:
        model.add_input(name='x_words_rand', input_shape=(None,), dtype='int')

    # input: discourse relation focus marking (doc, time_pad)
    model.add_input(name='x_rel_focus', input_shape=(max_len,), dtype='int')

    # shared 1: word embedding layer (doc, time_pad, emb)
    model.add_node(Embedding(words2id_size, embedding_dim, input_length=max_len, init='glorot_uniform', mask_zero=True), name='shared_1', input='x_words_pad')

    # shared N: bidirectional GRU full sequence layer (doc, time_pad, repr)
    shared_prev = 'shared_1'
    for n in range(2, shared_layers + 1):
        shared_fwd = 'shared_{}_fwd'.format(n)
        shared_bck = 'shared_{}_bck'.format(n)
        shared_join = 'shared_{}'.format(n)
        model.add_node(GRU(embedding_dim, return_sequences=True, activation='sigmoid', inner_activation='sigmoid', init='he_uniform', inner_init='orthogonal'), name=shared_fwd, input=shared_prev)
        model.add_node(GRU(embedding_dim, return_sequences=True, activation='sigmoid', inner_activation='sigmoid', init='he_uniform', inner_init='orthogonal', go_backwards=True), name=shared_bck, input=shared_prev)
        model.add_node(TimeDistributedDense(embedding_dim, init='he_uniform'), name=shared_join, inputs=[shared_prev, shared_fwd, shared_bck], merge_mode='concat')
        shared_prev = shared_join

    # model: skip-gram labels (sample, time_pad, offset)
    if 'x_skipgram' in loss:
        skipgram_out = None  #XXX:skipgram_model(model, ['shared_1', 'x_words_rand'], max_len, embedding_dim, words2id_size, skipgram_offsets)
        model.add_output(name='x_skipgram', input=skipgram_out)

    # model: POS tags as one-hot vectors (sample, time_pad, pos_tags2id)
    if 'x_pos_tags' in loss:
        pos_tags_out = pos_tags_model(model, ['shared_2'], max_len, embedding_dim, pos_tags2id_size)
        model.add_output(name='x_pos_tags', input=pos_tags_out)

    # model: discourse relation types as normalized vectors (sample, time_pad, rel_types2id)
    if 'x_rel_types' in loss:
        rel_types_out = rel_types_model(model, ['shared_2'], max_len, embedding_dim, rel_types2id_size, 'x_rel_focus')
        model.add_output(name='x_rel_types', input=rel_types_out)

    # model: discourse relation senses as normalized vectors (sample, time_pad, rel_senses2id)
    if 'x_rel_senses' in loss:
        rel_senses_out = rel_senses_model(model, ['shared_2'], max_len, embedding_dim, rel_senses2id_size, 'x_rel_focus')
        model.add_output(name='x_rel_senses', input=rel_senses_out)

    model.compile(optimizer='rmsprop', loss=loss)
    return model


### Prepare data

def relation_sample(rel_id, word_crop, max_len, doc_ids, words, word_metas, pos_tags, dependencies, parsetrees, rel_ids, rel_parts, rel_types, rel_senses, words2id, words2id_size, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size):
    doc_id = rel_parts[rel_id]['DocID']
    words_len = len(words[doc_id])
    token_min = rel_parts[rel_id]['TokenMin']
    token_max = rel_parts[rel_id]['TokenMax']
    rel_tag = rtsip_to_tag(rel_types[rel_id], rel_senses[rel_id], rel_id, "")

    # determine token boundaries around each relation
    if token_max - token_min < word_crop:
        token_start = token_min - (word_crop - (token_max - token_min) + 1) // 2
    else:
        token_start = token_min
    if token_start < 0:
        token_start = 0
    token_end = token_start + word_crop
    if token_end > words_len:
        token_start -= token_end - words_len
        if token_start < 0:
            token_start = 0
        token_end = token_start + word_crop

    # prepare data
    words_slice = words[doc_id][token_start:token_end]
    x1_words_pad = encode_x_words(words_slice, words2id, words2id_size, max_len)
    x1_words_rand = encode_x_words_rand(words_slice, words2id, words2id_size, max_len)

    x1_skipgram = None

    pos_tags_slice = pos_tags[doc_id][token_start:token_end]
    x1_pos_tags = encode_x_pos_tags(pos_tags_slice, pos_tags2id, pos_tags2id_size, max_len)

    word_metas_slice = word_metas[doc_id][token_start:token_end]
    x1_rel_types = encode_x_rel_types(word_metas_slice, rel_types2id, rel_types2id_size, max_len, filter_prefixes=[rel_tag])
    x1_rel_senses = encode_x_rel_senses(word_metas_slice, rel_senses2id, rel_senses2id_size, max_len, filter_prefixes=[rel_tag])
    x1_rel_focus = encode_x1_rel_focus(word_metas_slice, max_len, filter_prefixes=[rel_tag])
    return x1_words_pad, x1_words_rand, x1_skipgram, x1_pos_tags, x1_rel_types, x1_rel_senses, x1_rel_focus, token_start, token_end


def batch_generator(word_crop, max_len, batch_size, doc_ids, words, word_metas, pos_tags, dependencies, parsetrees, rel_ids, rel_parts, rel_types, rel_senses, words2id, words2id_size, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size):
    """Batch generator where each sample represents a different discourse relation."""

    rel_ids = rel_ids[:]
    while True:
        # shuffle relations on each epoch
        random.shuffle(rel_ids)

        for batch_start, batch_end in make_batches(len(rel_ids), batch_size):

            # prepare batch data
            x_words_pad = []
            x_words_rand = []
            x_skipgram = []
            x_pos_tags = []
            x_rel_types = []
            x_rel_senses = []
            x_rel_focus = []
            for rel_id in rel_ids[batch_start:batch_end]:
                x1_words_pad, x1_words_rand, x1_skipgram, x1_pos_tags, x1_rel_types, x1_rel_senses, x1_rel_focus, token_start, token_end = relation_sample(rel_id, word_crop, max_len, doc_ids, words, word_metas, pos_tags, dependencies, parsetrees, rel_ids, rel_parts, rel_types, rel_senses, words2id, words2id_size, pos_tags2id, pos_tags2id_size, rel_types2id, rel_types2id_size, rel_senses2id, rel_senses2id_size, rel_marking2id, rel_marking2id_size)
                x_words_pad.append(x1_words_pad)
                x_words_rand.append(x1_words_rand)
                x_skipgram.append(x1_skipgram)
                x_pos_tags.append(x1_pos_tags)
                x_rel_types.append(x1_rel_types)
                x_rel_senses.append(x1_rel_senses)
                x_rel_focus.append(x1_rel_focus)

            # yield batch
            yield {
                'x_words_pad': np.asarray(x_words_pad, dtype=np.int),
                #'x_words_rand': np.asarray(x_words_rand, dtype=np.int),
                'x_rel_focus': np.asarray(x_rel_focus, dtype=np.int),
                #'x_skipgram': np.asarray(x_skipgram, dtype=np.float32),
                #'x_pos_tags': np.asarray(x_pos_tags, dtype=np.float32),
                'x_rel_types': np.asarray(x_rel_types, dtype=np.float32),
                'x_rel_senses': np.asarray(x_rel_senses, dtype=np.float32),
            }


### Tests

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
