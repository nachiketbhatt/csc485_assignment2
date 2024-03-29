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
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = Counter([wsd.wordform for wsd in sentence])
    norm_context = 0
    for key in context.keys():
        norm_context += context[key] ** 2
    for synset in wn.synsets(sentence[word_index].lemma):
        signature = Counter()
        definition = synset.definition()
        examples = synset.examples()
        signature = signature + Counter(stop_tokenize(definition))
        for example in examples:
            signature = signature.__add__(Counter(stop_tokenize(example)))
        for hypo in synset.hyponyms():
            definition = hypo.definition()
            examples = hypo.examples()
            signature = signature.__add__(Counter(stop_tokenize(definition)))
            for example in examples:
                signature = signature.__add__(Counter(stop_tokenize(example)))
        for hypo in synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms():
            definition = hypo.definition()
            examples = hypo.examples()
            signature = signature.__add__(Counter(stop_tokenize(definition)))
            for example in examples:
                signature = signature.__add__(Counter(stop_tokenize(example)))
        for hypo in synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms():
            definition = hypo.definition()
            examples = hypo.examples()
            signature = signature.__add__(Counter(stop_tokenize(definition)))
            for example in examples:
                signature = signature.__add__(Counter(stop_tokenize(example)))
        score = 0
        dot_prodcut = 0
        for key in context.keys():
            if key in signature:
                dot_prodcut += context[key] * signature[key]
        norm_sig = 0
        for key in context.keys():
            norm_sig += signature[key] ** 2
        if (norm_sig * norm_context) != 0:
            score = dot_prodcut / ((norm_sig * norm_context) ** 0.5)
        if score > best_score:
            best_score = score
            best_sense = synset
    return best_sense


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
    v, d = word2vec.shape
    best_sense = mfs(sentence, word_index)
    best_score = 0
    context = np.zeros(d)
    for word in [wsd.wordform for wsd in sentence]:
        context += _get_word_vec(word, vocab, word2vec)
    context = context / len(sentence)
    for synset in wn.synsets(sentence[word_index].lemma):
        signature = np.zeros(d)
        definition = synset.definition()
        examples = synset.examples()
        count = 0
        for word in stop_tokenize(definition):
            signature += _get_word_vec(word, vocab, word2vec)
            count += 1
        for example in examples:
            for word in stop_tokenize(example):
                signature += _get_word_vec(word, vocab, word2vec)
                count += 1
        for hypo in synset.hyponyms():
            definition = hypo.definition()
            examples = hypo.examples()
            for word in stop_tokenize(definition):
                signature += _get_word_vec(word, vocab, word2vec)
                count += 1
            for example in examples:
                for word in stop_tokenize(example):
                    signature += _get_word_vec(word, vocab, word2vec)
                    count += 1
        for hypo in synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms():
            definition = hypo.definition()
            examples = hypo.examples()
            for word in stop_tokenize(definition):
                signature += _get_word_vec(word, vocab, word2vec)
                count += 1
            for example in examples:
                for word in stop_tokenize(example):
                    signature += _get_word_vec(word, vocab, word2vec)
                    count += 1
        for hypo in synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms():
            definition = hypo.definition()
            examples = hypo.examples()
            for word in stop_tokenize(definition):
                signature += _get_word_vec(word, vocab, word2vec)
                count += 1
            for example in examples:
                for word in stop_tokenize(example):
                    signature += _get_word_vec(word, vocab, word2vec)
                    count += 1
        signature = signature / count
        score = 0
        if norm(signature) * norm(context) != 0:
            score = np.dot(context, signature) / (norm(signature) * norm(context))
        if score > best_score:
            best_score = score
            best_sense = synset
    return best_sense


def _get_word_vec(word, vocab, word2vec):
    v, d = word2vec.shape
    if word in vocab:
        return word2vec[vocab[word]]
    elif word.lower() in vocab:
        return word2vec[vocab[word.lower()]]
    else:
        return np.zeros(d)


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
