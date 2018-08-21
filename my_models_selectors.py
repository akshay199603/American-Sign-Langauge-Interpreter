# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:48:17 2018

@author: aksha
"""

import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM

# Base class model selection
class ModelSelector(object):
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

# Implement Selector Constant, select the model with value self.n_constant
class SelectorConstant(ModelSelector):
    def select(self):
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

# Implement the Bayesian Information Criterion (BIC), select the model with lowest BIC score
class SelectorBIC(ModelSelector):
    def score(self, num_states):
        model = self.base_model(n)
        features = len(self.X)
        logL = model.score(self.X, self.lengths)
        logN = np.log(features)
        parameters = num_states ** 2 + 2 * num_states * n - 1
        d = model.n_features
        p = num_states ** 2 + 2 * d * num_states - 1
        return -2.0 * logL + p * logN, model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            bics = []
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                bics.append(score(num_states))
            best_model = max(bics)[1]
            return best_model
        except:
            return self.base_model(self.n_constant)