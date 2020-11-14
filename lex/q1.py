#!/usr/bin/env python3
# Student name: Nachiket Bhatt
# Student number: 1004703332
# UTORid: bhattnac

from collections import Counter, defaultdict
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from tqdm import tqdm, trange

from q0 import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    syns = wn.synsets(sentence[word_index].lemma)
    return syns[0]


def lesk(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Simplified Lesk algorithm.
    raise NotImplementedError

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = set([wsd.wordform for wsd in sentence])
    for synset in wn.synsets(sentence[word_index].lemma):
        signature = set()
        definition = synset.definition()
        examples = synset.examples()
        signature = signature.union(set(stop_tokenize(definition)))
        for example in examples:
            signature = signature.union(set(stop_tokenize(example)))
        score = len(context.intersection(signature))
        if score > best_score:
            best_score = score
            best_sense = synset
    return best_sense


def lesk_ext(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = set([wsd.wordform for wsd in sentence])
    for synset in wn.synsets(sentence[word_index].lemma):
        signature = set()
        definition = synset.definition()
        examples = synset.examples()
        signature = signature.union(set(stop_tokenize(definition)))
        for example in examples:
            signature = signature.union(set(stop_tokenize(example)))
        for hypo in synset.hyponyms():
            definition = hypo.definition()
            examples = hypo.examples()
            signature = signature.union(set(stop_tokenize(definition)))
            for example in examples:
                signature = signature.union(set(stop_tokenize(example)))
        for hypo in synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms():
            definition = hypo.definition()
            examples = hypo.examples()
            signature = signature.union(set(stop_tokenize(definition)))
            for example in examples:
                signature = signature.union(set(stop_tokenize(example)))
        for hypo in synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms():
            definition = hypo.definition()
            examples = hypo.examples()
            signature = signature.union(set(stop_tokenize(definition)))
            for example in examples:
                signature = signature.union(set(stop_tokenize(example)))
        score = len(context.intersection(signature))
        if score > best_score:
            best_score = score
            best_sense = synset
    return best_sense


def lesk_cos(sentence: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    raise NotImplementedError


def lesk_w2v(sentence: Sequence[WSDToken], word_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    When looking up the vector, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    raise NotImplementedError


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
