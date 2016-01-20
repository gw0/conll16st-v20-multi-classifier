#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Load CoNLL16st/CoNLL15st dataset.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2016@tnode.com>"
__license__ = "GPLv3+"

import codecs
import json


def load_parses(dataset_dir, doc_ids=None, parses_ffmts=None):
    """Load parses and tags from CoNLL16st corpus."""
    if parses_ffmts is None:
        parses_ffmts = [
            "{}/parses.json",             # CoNLL16st filenames
            "{}/pdtb-parses.json",        # CoNLL15st filenames
            "{}/pdtb_trial_parses.json",  # CoNLL15st trial filenames
        ]

    # load parses
    parses = {}
    for parses_ffmt in parses_ffmts:
        try:
            f = codecs.open(parses_ffmt.format(dataset_dir), 'r', encoding='utf8')
            parses = json.load(f)
            f.close()
            break
        except IOError:
            pass

    # filter by document ids
    if doc_ids is not None:
        parses = { doc_id: parses[doc_id]  for doc_id in doc_ids }
    return parses


def load_raws(dataset_dir, doc_ids, raw_ffmts=None):
    """Load raw text by document id from CoNLL16st corpus."""
    if raw_ffmts is None:
        raw_ffmts = [
            "{}/raw/{}",  # CoNLL16st/CoNLL15st filenames
        ]

    # load raw texts
    raws = {}
    for doc_id in doc_ids:
        raws[doc_id] = None
        for raw_ffmt in raw_ffmts:
            try:
                f = codecs.open(raw_ffmt.format(dataset_dir, doc_id), 'r', encoding='utf8')
                raws[doc_id] = f.read()
                f.close()
                break  # skip other filenames
            except IOError:
                pass
    return raws


### Tests

def test_parses():
    dataset_dir = "./conll16st-en-trial"
    t_doc_id = "wsj_1000"
    t_s0_word0 = "Kemper"
    t_s0_word0_linkers = ["arg1_14890"]
    t_s0_word0_pos = "NNP"
    t_s0_parsetree = "( (S (NP (NNP Kemper) (NNP Financial) (NNPS Services)"
    t_s0_dependency0 = ["root", "ROOT-0", "cut-16"]
    t_s0_dependency1 = ["nn", "Inc.-4", "Kemper-1"]

    parses = load_parses(dataset_dir)
    s0 = parses[t_doc_id]['sentences'][0]
    assert s0['words'][0][0] == t_s0_word0
    assert s0['words'][0][1]['Linkers'] == t_s0_word0_linkers
    assert s0['words'][0][1]['PartOfSpeech'] == t_s0_word0_pos
    assert s0['parsetree'].startswith(t_s0_parsetree)
    assert t_s0_dependency0 in s0['dependencies']
    assert t_s0_dependency1 in s0['dependencies']

def test_raws():
    dataset_dir = "./conll16st-en-trial"
    doc_id = "wsj_1000"
    t_raw = ".START \n\nKemper Financial Services Inc., charging"

    raws = load_raws(dataset_dir, [doc_id])
    assert raws[doc_id].startswith(t_raw)

if __name__ == '__main__':
    import pytest
    pytest.main(['-s', __file__])
