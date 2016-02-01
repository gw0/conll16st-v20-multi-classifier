#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Load CoNLL16st/CoNLL15st dataset.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

from files import load_parses, load_raws, load_relations_gold
from words import get_words, get_pos_tags, get_word_metas
from dependencies import get_dependencies
from parsetrees import get_parsetrees
from relations import get_rel_parts, get_rel_types, get_rel_senses, add_relation_tags


def load_all(dataset_dir, doc_ids=None, filter_types=None, filter_senses=None):
    """Load whole CoNLL16st dataset by document id."""

    # load all provided files untouched
    parses = load_parses(dataset_dir, doc_ids=doc_ids)
    doc_ids = sorted(parses.keys())
    raws = load_raws(dataset_dir, doc_ids=doc_ids)
    relationsnos_gold = load_relations_gold(dataset_dir, doc_ids=doc_ids, with_senses=False, filter_types=filter_types, filter_senses=filter_senses)
    relations_gold = load_relations_gold(dataset_dir, doc_ids=doc_ids, with_senses=True, filter_types=filter_types, filter_senses=filter_senses)

    # extract data by document id and token id
    words = get_words(parses)
    pos_tags = get_pos_tags(parses)
    word_metas = get_word_metas(parses, raws)

    # extract data by document id and token id pairs
    dependencies = get_dependencies(parses)

    # extract data by document id
    parsetrees = get_parsetrees(parses)

    # extract data by relation id
    rel_parts = get_rel_parts(relationsnos_gold)
    rel_types = get_rel_types(relations_gold)
    rel_senses = get_rel_senses(relations_gold)
    rel_ids = rel_parts.keys()

    # add extra fields
    add_relation_tags(word_metas, rel_types, rel_senses)

    return doc_ids, words, word_metas, pos_tags, dependencies, parsetrees, rel_ids, rel_parts, rel_types, rel_senses, relations_gold
