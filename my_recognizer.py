# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:54:51 2018

@author: aksha
"""

import warnings
from asl_data import SinglesData

# Recognize the test word sequences from the word set
def recognize(models: dict, test_set: SinglesData):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    for word_id, (X, lengths) in test_set.get_all_Xlengths().items():
        word_prob = {}
        for model_key, model_value in models.items():
            try:
                score = model_value.score(X, lengths)
                word_prob[model_key] = score
            except:
                word_prob[model_key] = float("-inf")
        probabilities.append(word_prob)
        guesses.append(max(word_prob, key=word_prob.get))
    return probabilities, guesses
    