# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:31:50 2018

@author: aksha
"""

import os
import numpy as np
import pandas as pd


class AslDb(object):
    
# Create the database by taking the input csv files with the hand position information, frame and speaker information
    def __init__(self,
                 hands_fn=os.path.join('data', 'hands_condensed.csv'),
                 speakers_fn=os.path.join('data', 'speaker.csv'),
                 ):
        self.df = pd.read_csv(hands_fn).merge(pd.read_csv(speakers_fn),on='video')
        self.df.set_index(['video','frame'], inplace=True)

# Takes the training words csv file as input and creates the data objects for training the words for hmmlearn library
    def build_training(self, feature_list, csvfilename =os.path.join('data', 'train_words.csv')):
        return WordsData(self, csvfilename, feature_list)

# Takes the test words csv file as input and creates objects for individual test words
    def build_test(self, feature_method, csvfile=os.path.join('data', 'test_words.csv')):
        return SinglesData(self, csvfile, feature_method)


class WordsData(object):
    
# Loads the taining data suitable to the method using hmmlearn library
    def __init__(self, asl:AslDb, csvfile:str, feature_list:list):
        self._data = self._load_data(asl, csvfile, feature_list)
        self._hmm_data = create_hmmlearn_data(self._data)
        self.num_items = len(self._data)
        self.words = list(self._data.keys())

# Creates the dictionary of words
    def _load_data(self, asl, fn, feature_list):
        tr_df = pd.read_csv(fn)
        dict = {}
        for i in range(len(tr_df)):
            word = tr_df.ix[i,'word']
            video = tr_df.ix[i,'video']
            new_sequence = [] 
            for frame in range(tr_df.ix[i,'startframe'], tr_df.ix[i,'endframe']+1):
                vid_frame = video, frame
                sample = [asl.df.ix[vid_frame][f] for f in feature_list]
                if len(sample) > 0: 
                    new_sequence.append(sample)
            if word in dict:
                dict[word].append(new_sequence) 
            else:
                dict[word] = [new_sequence]
        return dict

# Find the sequences of feature lists for each frame
    def get_all_sequences(self):
        return self._data

# Get the data in the form a (X, lengths) suitable with hmmlearn library
    def get_all_Xlengths(self):
        return self._hmm_data


    def get_word_sequences(self, word:str):
        return self._data[word]

    def get_word_Xlengths(self, word:str):
        return self._hmm_data[word]


class SinglesData(object):
    
# Loads the taining data suitable to the method using hmmlearn library
    def __init__(self, asl:AslDb, csvfile:str, feature_list):
        self.df = pd.read_csv(csvfile)
        self.wordlist = list(self.df['word'])
        self.sentences_index  = self._load_sentence_word_indices()
        self._data = self._load_data(asl, feature_list)
        self._hmm_data = create_hmmlearn_data(self._data)
        self.num_items = len(self._data)
        self.num_sentences = len(self.sentences_index)
# Consolidates sequenced feature data into a dictionary of words and creates answer list
    def _load_data(self, asl, feature_list):
        dict = {}
        for i in range(len(self.df)):
            video = self.df.ix[i,'video']
            new_sequence = [] 
            for frame in range(self.df.ix[i,'startframe'], self.df.ix[i,'endframe']+1):
                vid_frame = video, frame
                sample = [asl.df.ix[vid_frame][f] for f in feature_list]
                if len(sample) > 0: 
                    new_sequence.append(sample)
            if i in dict:
                dict[i].append(new_sequence)
            else:
                dict[i] = [new_sequence]
        return dict

# create dict of video sentence numbers with list of word indices as values
        
    def _load_sentence_word_indices(self):
        working_df = self.df.copy()
        working_df['idx'] = working_df.index
        working_df.sort_values(by='startframe', inplace=True)
        p = working_df.pivot('video', 'startframe', 'idx')
        p.fillna(-1, inplace=True)
        p = p.transpose()
        dict = {}
        for v in p:
            dict[v] = [int(i) for i in p[v] if i>=0]
        return dict

    def get_all_sequences(self):
        return self._data

    def get_all_Xlengths(self):
        return self._hmm_data

    def get_item_sequences(self, item:int):
        return self._data[item]

    def get_item_Xlengths(self, item:int):
        return self._hmm_data[item]

# concatenates sequences and return tuple of the new list and lengths  
def combine_sequences(sequences):
    sequence_cat = []
    sequence_lengths = []
    for sequence in sequences:
        sequence_cat += sequence
        num_frames = len(sequence)
        sequence_lengths.append(num_frames)
    return sequence_cat, sequence_lengths

# Create the hmm learn data
def create_hmmlearn_data(dict):
    seq_len_dict = {}
    for key in dict:
        sequences = dict[key]
        sequence_cat, sequence_lengths = combine_sequences(sequences)
        seq_len_dict[key] = np.array(sequence_cat), sequence_lengths
    return seq_len_dict

if __name__ == '__main__':
    asl = AslDb()

