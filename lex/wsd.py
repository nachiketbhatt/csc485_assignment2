#!/usr/bin/env python3

from dataclasses import dataclass, field
from pathlib import Path
from typing import *

import gzip
import pickle

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


# set MODELS_DIR to None in order to run this outside of teach.cs
MODELS_DIR = Path('/u/csc485h/fall/pub/lms/')
#MODELS_DIR = None


@dataclass
class WSDToken:
    wordform: str
    lemma: str
    synsets: Set[str] = field(default_factory=set)


def load_bert(model: str = 'bert-base-cased'):
    tf_dir = MODELS_DIR / 'transformers' if MODELS_DIR else MODELS_DIR
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=tf_dir,
                                              use_fast=True)
    model = AutoModel.from_pretrained(model, cache_dir=tf_dir)
    return tokenizer, model


def _load_data(subset: str) -> List[List[WSDToken]]:
    with gzip.open(f'corpora/{subset}.pkl.gz') as data_in:
        return pickle.load(data_in)


def load_eval() -> List[List[WSDToken]]:
    return _load_data('eval')


def load_train() -> List[List[WSDToken]]:
    return _load_data('train')


def load_word2vec() -> Tuple[Mapping[str, int], np.ndarray]:
    w2v_dir = MODELS_DIR if MODELS_DIR else Path()
    with gzip.open(w2v_dir / 'w2v.vocab.pkl.gz') as vocab_in:
        vocab = {w: i for i, w in enumerate(pickle.load(vocab_in))}
    vectors = np.load(w2v_dir / 'w2v.npy')

    return vocab, vectors


def evaluate(corpus: List[List[WSDToken]], wsd_func: Callable[..., Synset],
             *func_args, **func_kwargs) -> Optional[float]:
    try:
        correct, total = 0, 0
        for sentence in tqdm(corpus, desc=(fn := wsd_func.__name__),
                             leave=False):
            for i, token in enumerate(sentence):
                if token.synsets and len(wn.synsets(token.lemma)) > 1:
                    predicted_sense = wsd_func(sentence, i, *func_args,
                                               **func_kwargs)
                    if predicted_sense.name() in token.synsets:
                        correct += 1
                    total += 1

        tqdm.write(f'{fn}: {(acc := correct / total):.1%}')
        return acc
    except NotImplementedError:
        return None
