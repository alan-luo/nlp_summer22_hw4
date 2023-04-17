#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # part 6

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
        # Part 4
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
    
    def predict_with_score(self,context : Context) -> "tuple[str, float]":
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

        return (best_lem, max_sim)


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        ctx = context.left_context + ['[MASK]'] + context.right_context
        ctx_str = ""
        for tok in ctx:
            if tok in [',', '.', ';', ':']:
                ctx_str = ctx_str + tok
            else:
                ctx_str = ctx_str + ' ' + tok

        mask_id = self.tokenizer.encode('[MASK]')[1]

        input_toks = self.tokenizer.encode(ctx_str)
        target_idx = input_toks.index(mask_id)

        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat, verbose = 0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][target_idx])[::-1]

        best_list = self.tokenizer.convert_ids_to_tokens(best_words)

        lowest_idx = float('inf')
        lowest_word = None

        for candidate in candidates:
            if candidate in best_list:
                idx = best_list.index(candidate)
                if (idx < lowest_idx):
                    lowest_idx = idx
                    lowest_word = candidate

        if (lowest_word == None):
            return wn_frequency_predictor(context)


        return lowest_word
    
    def predict_with_lemmas(self, context : Context) -> str:
        # Part 6 - I thought it was weird that in the candidate list we were passing in lemmas
        # while the outputs from BERT are in the conjugated / modified word forms. So, in this
        # approach I thought of lemmatizing all of the BERT outputs before looking at the
        # candidate list. Unfortunately it actually gives worse performance than the raw BERT
        # output. Not sure why.
        wnl = WordNetLemmatizer()

        candidates = get_candidates(context.lemma, context.pos)
        ctx = context.left_context + ['[MASK]'] + context.right_context
        ctx_str = ""
        for tok in ctx:
            if tok in [',', '.', ';', ':']:
                ctx_str = ctx_str + tok
            else:
                ctx_str = ctx_str + ' ' + tok

        mask_id = self.tokenizer.encode('[MASK]')[1]

        input_toks = self.tokenizer.encode(ctx_str)
        target_idx = input_toks.index(mask_id)

        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat, verbose = 0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][target_idx])[::-1]

        best_list = self.tokenizer.convert_ids_to_tokens(best_words)
        best_list = [wnl.lemmatize(w) for w in best_list]

        lowest_idx = float('inf')
        lowest_word = None

        for candidate in candidates:
            if candidate in best_list:
                idx = best_list.index(candidate)
                if (idx < lowest_idx):
                    lowest_idx = idx
                    lowest_word = candidate

        if (lowest_word == None):
            return wn_frequency_predictor(context)


        return lowest_word 
    
    def predict_with_score(self, context : Context) -> "tuple[str, float]":
        candidates = get_candidates(context.lemma, context.pos)
        ctx = context.left_context + ['[MASK]'] + context.right_context
        ctx_str = ""
        for tok in ctx:
            if tok in [',', '.', ';', ':']:
                ctx_str = ctx_str + tok
            else:
                ctx_str = ctx_str + ' ' + tok

        mask_id = self.tokenizer.encode('[MASK]')[1]

        input_toks = self.tokenizer.encode(ctx_str)
        target_idx = input_toks.index(mask_id)

        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat, verbose = 0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][target_idx])[::-1]

        best_list = self.tokenizer.convert_ids_to_tokens(best_words)

        lowest_idx = float('inf')
        lowest_word = None

        for candidate in candidates:
            if candidate in best_list:
                idx = best_list.index(candidate)
                if (idx < lowest_idx):
                    lowest_idx = idx
                    lowest_word = candidate

        if (lowest_word == None):
            return wn_frequency_predictor(context)

        score = (1 - (lowest_idx / len(best_list))) ** 100

        return (lowest_word, score)

# w2v and bert have surprisingly similar scores. Inspecting the .predict files, it seems like 
# there's a chance that these hits come from non-identical sets. That is, if we can somehow
# figure out which context should use which predictor, we might be able to get a score of up to
# 34% = 2 * 17% (in the very optimistic case when these sets are disjoint). We'll throw in the
# baseline predictor into the mix for good measure.

# The way this works: if the score from w2v is not good enough, use BERT. If the score from BERT
# is not good enough, use baseline.

# With this very convoluted method we managed to improve our score by 0.5% ;)
def w2v_bert_predict(context, predictor, b_predictor):
    THRESH_1 = 0.5
    THRESH_2 = 0.85

    w2v_pred = predictor.predict_with_score(context)
    bert_pred = b_predictor.predict_with_score(context)

    # print(w2v_pred, bert_pred)

    if (w2v_pred[1] > THRESH_1):
        return w2v_pred[0]
    
    if (bert_pred[1] > THRESH_2):
        return bert_pred[0]
    
    return wn_frequency_predictor(context)

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = '~/GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    b_predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging

        # prediction = wn_frequency_predictor(context)          # baseline.predict
        # prediction = wn_simple_lesk_predictor(context)        # lesk.predict 
        # prediction = predictor.predict_nearest(context)       # w2v.predict
        # prediction = b_predictor.predict(context)             # bert.predict
        # prediction = b_predictor.predict_with_lemmas(context) # bertlem.predict
        prediction = w2v_bert_predict(context, predictor, b_predictor) # fancy.predict

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
