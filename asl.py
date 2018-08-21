# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:32:25 2018

@author: aksha
"""

import numpy as np
from my_models_selectors import SelectorConstant
import timeit
from my_recognizer import recognize
from asl_utils import show_errors
from my_models_selectors import SelectorBIC
from asl_data import AslDb
import warnings
from hmmlearn.hmm import GaussianHMM
from asl_utils import test_std_tryit

# upload the database created using asl_data.py file
asl = AslDb()
asl.df.head()

# Find the values for ground truth
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
training = asl.build_training(features_ground)

# Get the training words from the database file
print("Training words: {}".format(training.words))
df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df.head()
df_std = asl.df[['left-x', 'left-y', 'right-x', 'right-y','nose-x','nose-y','speaker','grnd-ry','grnd-rx','grnd-ly','grnd-lx']].groupby('speaker').std()
#test_std_tryit(df_std)

# Find the features used for training
asl.df['norm-lx']= (asl.df['left-x']-asl.df['speaker'].map(df_means['left-x']))/ asl.df['speaker'].map(df_std['left-x'])
asl.df['norm-ly']= (asl.df['left-y']-asl.df['speaker'].map(df_means['left-y']))/ asl.df['speaker'].map(df_std['left-y'])
asl.df['norm-rx']= (asl.df['right-x']-asl.df['speaker'].map(df_means['right-x']))/ asl.df['speaker'].map(df_std['right-x'])
asl.df['norm-ry']= (asl.df['right-y']-asl.df['speaker'].map(df_means['right-y']))/ asl.df['speaker'].map(df_std['right-y'])

features_normal = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']


asl.df['polar-rr']= (asl.df['grnd-rx'].apply(np.square)+asl.df['grnd-ry'].apply(np.square)).apply(np.sqrt)
asl.df['polar-rtheta']=  np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr']= (asl.df['grnd-lx'].apply(np.square)+asl.df['grnd-ly'].apply(np.square)).apply(np.sqrt)
asl.df['polar-ltheta']= np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])

features_polar_coordinates = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

# Train a specific word
def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'Cop'
model, logL = train_a_word(demoword, 3, features_ground)

# Find the hidden states of the words
def show_model_stats(word, model):
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  
        print("hidden state #{}".format(i))
        print()
    
show_model_stats(demoword, model)

# Include the words to train
words_to_train = training.words

# Train the first model Bayesian Information Criterion
training = asl.build_training(features_ground)
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))

# Train all the training words found from the dataset            
def train_all_words(features, model_selector):
    training = asl.build_training(features)  
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorConstant)
print("Number of word models returned = {}".format(len(models)))

test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))

features_to_try = {"features_polar":features_polar_coordinates}
model_selectors_to_try = {"SelectorBIC":SelectorBIC}
feature_models = [(features_to_try[features], model_selectors_to_try[model_selector], features, model_selector) for features in features_to_try.keys() for model_selector in model_selectors_to_try]

# Test the model
for f, m, f_name, m_name in feature_models:
    models = train_all_words(f, m)
    test_set = asl.build_test(f)
    probabilities, guesses = recognize(models, test_set)
    show_errors(guesses, test_set)