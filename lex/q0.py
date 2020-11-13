#!/usr/bin/env python3
# Student name: Nachiket Bhatt
# Student number: 1004703332
# UTORid: bhattnac

from string import punctuation
from typing import *

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize

def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    max_depth_synset = max(wn.all_synsets(), key=lambda synset: synset.max_depth())
    print(max_depth_synset)


def superdefn(s: str) -> List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up...)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        s (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    definitions = wn.synset(s).definition()
    for synset in wn.synset('toughen.v.01').hypernyms():
        definitions += ' ' + synset.definition()
    for synset in wn.synset('toughen.v.01').hyponyms():
        definitions += ' ' + synset.definition()
    return word_tokenize(definitions)


def stop_tokenize(s: str) -> List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    filtered = []
    stop_words = set(stopwords.words("english"))
    punc_words = set(punctuation.words("english"))
    for word in word_tokenize(s):
        if not (word in stop_words or word in punc_words):
            filtered.append(word)
    return filtered


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    deepest()
