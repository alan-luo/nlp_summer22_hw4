#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    lemmas = set()
    synsets = wn.synsets(lemma, pos=pos)
    for synset in synsets:
        lemmas.update(synset.lemmas())

    word_set = set(map(lambda l: l.name().replace('_', ' '), lemmas))
    word_set.remove(lemma)

    return list(word_set)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    # Get a map from all lemmas to counts, pick the most common lemma

    # Construct a map from lemmas to counts
    lem_to_count = dict()

    synsets = wn.synsets(context.lemma, pos=context.pos)
    for synset in synsets:
        for lemma in synset.lemmas():
            name = lemma.name()
            if (name == context.lemma):
                continue
            if (name in lem_to_count):
                lem_to_count[name] += lemma.count()
            else:
                lem_to_count[name] = lemma.count()

    max_count = -1
    max_lem = None

    for lem, count in lem_to_count.items():
        if count > max_count:
            max_count = count
            max_lem = lem


    return max_lem.replace('_', ' ')

# Get the definition and examples of a synset, remove all stopwords, concat into a big array 
def get_syn_gloss(synset, remove_stops): # str[]
    syn_def = list(filter(remove_stops, synset.definition().split()))

    syn_examples = []
    for example in synset.examples():
        syn_examples = syn_examples + list(filter(remove_stops, example.split()))

    syn_gloss = syn_def + syn_examples
    return syn_gloss
    
def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3

    # Select best synset by highest gloss overlap, then select most freq in that synset
    
    stop_words = stopwords.words('english')
    stop_words = stop_words + [',', '.', ';', ':']
    remove_stops = lambda word: word not in stop_words


    # Get word context
    word_ctx = context.left_context + context.right_context
    word_ctx = list(filter(remove_stops, word_ctx))

    # Compute highest overlapping synsets
    max_intersect = -1
    best_synsets = []

    synsets = wn.synsets(context.lemma, pos=context.pos)
    for synset in synsets:
        syn_gloss = get_syn_gloss(synset, remove_stops) 

        for hyp_syn in synset.hypernyms():
            hyp_gloss = get_syn_gloss(hyp_syn, remove_stops)
            syn_gloss = syn_gloss + hyp_gloss

        intersect = list(filter(lambda word: word in word_ctx, syn_gloss))

        if len(intersect) > max_intersect:
            max_intersect = len(intersect)
            best_synsets = [synset]
        elif len(intersect) == max_intersect:
            best_synsets.append(synset)

    max_syn_count = -1
    best_syn_lemma = None # basically best_best_lemma lol

    # Pick the synset->lemma pair with the highest freq
    for synset in best_synsets:
        # best_lemma is the most freq lemma in this synset
        max_count = -1
        best_lemma = None
        for lemma in synset.lemmas():
            if lemma.count() > max_count:
                max_count = lemma.count()
                best_lemma = lemma

        # now, compare with our global best
        if max_count > max_syn_count:
            max_syn_count = max_count
            best_syn_lemma = best_lemma

    return best_syn_lemma.name().replace('_', ' ')


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        max_sim = -1
        best_lem = None
        for candidate in candidates:
            try:
                sim = self.model.similarity(context.lemma, candidate)
                if (sim > max_sim):
                    max_sim = sim
                    best_lem = candidate
            except KeyError:
                continue

        if best_lem == None:
            return wn_frequency_predictor(context)

        return best_lem


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        return None # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = '~/GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        prediction = predictor.predict_nearest(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
