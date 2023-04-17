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

    word_set = set(map(lambda l: l.name(), lemmas))
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


    return max_lem

def wn_simple_lesk_predictor(context : Context) -> str:
    return None #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        return None # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        return None # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        print(context)  # useful for debugging
        prediction = wn_frequency_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
