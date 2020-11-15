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

import torch

from tqdm import tqdm, trange

from transformers import PreTrainedTokenizerFast, PreTrainedModel

from q1 import mfs
from wsd import evaluate, load_bert, load_eval, load_train, WSDToken


def gather_sense_vectors(corpus: List[List[WSDToken]],
                         tokenizer: PreTrainedTokenizerFast,
                         bert_model: PreTrainedModel,
                         batch_size: int = 16) \
        -> Dict[Synset, np.ndarray]:
    """Gather sense vectors using BERT run over a corpus.

    As with A1, it is much more efficient to batch the sentences up than it is
    to do one sentence at a time, and you can further improve (~twice as fast)
    if you sort the corpus by sentence length first. We've therefore started
    this function out that way for you, but you may implement the code in this
    function however you like.

    The procedure for this function is as follows:
    * Run bert_model on each batch (more information below).
    * Go through all of the WSDTokens in the input batch. For each one, if the
      token has any synsets assigned to it (check WSDToken.synsets), then add
      the BERT output vector to a list of vectors for that sense (**not** for
      the token!).
    * Once this is done for all batches, then for each synset that was seen
      in the corpus, compute the mean of all vectors stored in its list.
    * That yields a single vector associated to each synset; return this as
      a dictionary.

    To run bert_model, first you must tokenize the inputs into the the
    tokenization that BERT wants. Because the sentence has already been split
    into words, you will need to use the `is_split_into_words` argument; and
    since you will be running this in batches, the batches must be made to have
    equal length using the `padding` argument; and since you will be passing
    the tokenized outputs to bert_model, `return_tensors` will be useful.
    You can then pass the result to bert_model. See the examples below.

    An important point: the tokenizer will produce more tokens than in the
    original input, because sometimes it will split one word into multiple
    pieces. bert_model will then produce one vector per token. In order to
    produce a single vector for each *original* word token, so that you can
    then use that vector for its various synsets, you will need to align the
    output tokens back to the originals. You will then sometimes have multiple
    vectors for a single token in the input data; take the mean of these to
    yield a single vector per token. This vector can then be used like any
    other in the procedure described above. The return_offsets_mapping argument
    for tokenizer will help with this; see the example below.

    Args:
        corpus (list of list of WSDToken): The corpus to use.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the BERT model.
        bert_model (PreTrainedModel):  The BERT model.
        batch_size (int): The batch size to use.

    Returns:
        dict mapping Synsets to np.ndarray: A dictionary that can be used to
        retrieve the (NumPy) vector for a given sense.

    Examples:
        >>> tokenizer('This is definitely a sentence.')
        {'input_ids': [101, 1188, 1110, 5397, 170, 5650, 119, 102],
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

        >>> t = tokenizer([['Multiple', ',', 'pre-tokenized', 'sentences', '!'],\
                           ['Much', 'wow', '!']],\
                          is_split_into_words=True, padding=True,\
                          return_tensors='pt', return_offsets_mapping=True)
        >>> tokenizer.convert_ids_to_tokens(t['input_ids'][0].tolist())
        ['[CLS]', 'Multiple', ',', 'pre', '-', 'token', '##ized', 'sentences',
         '!', '[SEP]']
        >>> tokenizer.convert_ids_to_tokens(t['input_ids'][1].tolist())
        ['[CLS]', 'Much', 'w', '##ow', '!', '[SEP]', '[PAD]', '[PAD]', '[PAD]',
        '[PAD]']
        >>> offset_mapping = t.pop('offset_mapping').tolist()
        >>> bert_model(**t)[0].shape
        torch.Size([2, 10, 768])
        >>> offset_mapping  # this can be used to align to the original tokens
        [[[0, 0], [0, 8], [0, 1], [0, 3], [3, 4], [4, 9], [9, 13], [0, 9],
         [0, 1], [0, 0]],
         [[0, 0], [0, 4], [0, 1], [1, 3], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]]]
    """
    corpus = sorted(corpus, key=len)
    dic = {}
    for batch_n in range(0, len(corpus), batch_size):
        if batch_n + batch_size < len(corpus):
            batch = corpus[batch_n:batch_n + batch_size]
        else:
            batch = corpus[batch_n:len(corpus)]
        words = [[wsd.wordform for wsd in sentence] for sentence in batch]
        tokens = tokenizer(words, is_split_into_words=True, padding=True,
                           return_tensors='pt', return_offsets_mapping=True)
        offset_mapping = tokens.pop('offset_mapping').tolist()
        vectors = bert_model(**tokens)[0]
        offset_align = []
        for mapping in offset_mapping:
            i, j = 0, 0
            ranges = []
            while i < len(mapping):
                if mapping[i] == [0, 0]:
                    i += 1
                    continue
                j = i + 1
                while j < len(mapping) and mapping[j][0] != 0:
                    j += 1
                ranges.append((i, j))
                i = j
            offset_align.append(ranges)
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                interval = offset_align[i][j]
                for syn in batch[i][j].synsets:
                    synset = wn.synset(syn)
                    if synset not in dic:
                        dic[synset] = np.mean(vectors[i][interval[0]:interval[1]], axis=1)
                    else:
                        dic[synset] = np.append(dic[synset], np.mean(vectors[i][interval[0]:interval[1]], axis=1), axis=0)
    for key in dic.keys():
        dic[key] = np.mean(dic[key], axis=1)
    return dic


def bert_1nn(sentence: Sequence[WSDToken], word_index: int,
             tokenizer: PreTrainedTokenizerFast, bert_model: PreTrainedModel,
             sense_vectors: Mapping[Synset, np.ndarray]) -> Synset:
    """Find the best sense for a word in a sentence using the most
    cosine-similar sense vector.

    See the docstring for gather_sense_vectors above for examples of how to use
    BERT. You will need to run BERT on input sentence and associate a single
    vector for each input token in the same way. Once you've done this, you can
    compare the vector for the target word with the sense vectors for its
    possible senses, and then return the sense with the highest cosine
    similarity.

    In case none of the senses have vectors, return the most frequent sense
    (e.g., by just calling mfs(), which has been imported from q1 for you).

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the BERT model.
        bert_model (PreTrainedModel):  The BERT model.
        sense_vectors: A dictionary mapping synsets to NumPy vectors,
            as generated by gather_sense_vectors(...).

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sentence, word_index)
    best_score = 0
    words = [wsd.wordform for wsd in sentence]
    tokens = tokenizer(words, is_split_into_words=True, padding=True,
                          return_tensors='pt', return_offsets_mapping=True)
    offset_mapping = tokens.pop('offset_mapping').tolist()[0]
    context_vector = bert_model(**tokens)[0]
    i = 0
    ranges = []
    while i < len(offset_mapping):
        if offset_mapping[i] == [0, 0]:
            i += 1
            continue
        j = i + 1
        while j < len(offset_mapping) and offset_mapping[j][0] != 0:
            j += 1
        ranges.append((i, j))
        i = j
    context_vector = context_vector[0][ranges[word_index][0]:ranges[word_index][1]]
    for synset in wn.synsets(sentence[word_index].lemma):
        score = 0
        if synset in sense_vectors:
            signature = sense_vectors[synset]
            score = np.dot(context_vector, signature) / (norm(context_vector) * norm(signature))
        if score > best_score:
            best_score = score
            best_sense = synset
    return best_sense


if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')

    with torch.no_grad():
        bert_tok, bert = load_bert()
        train_data = load_train()
        eval_data = load_eval()

        sense_vecs = gather_sense_vectors(train_data, bert_tok, bert)
        evaluate(eval_data, bert_1nn, bert_tok, bert, sense_vecs)
