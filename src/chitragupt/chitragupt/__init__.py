'''
chitragoopt: Main module

Copyright 2015, Venkatesh Rao
Licensed under MIT.
'''

import nltk.metrics
from nltk.corpus import movie_reviews
from featx import label_feats_from_corpus, split_label_feats
from nltk.classify import NaiveBayesClassifier
from featx import bag_of_words
from nltk.classify.util import accuracy
import collections
import pickle
import sys

def word_feats(words):
    return dict([(word, True) for word in words])


def main():
    '''
    Main function of the boilerplate code is the entry point of the 'chitragoopt' executable script (defined in setup.py).
    
    Use doctests, those are very helpful.
    
    >>> main()
    Hello
    >>> 2 + 2
    4
    '''

    lfeats = label_feats_from_corpus(movie_reviews)
    train_feats, test_feats = split_label_feats(lfeats, split=0.75)
    train_feats, test_feats = split_label_feats(lfeats, split=0.75)
    # nb_classifier = NaiveBayesClassifier.train(train_feats)
    print(sys.argv[1].split())
    negfeat = bag_of_words(sys.argv[1].split())

    f = open('my_classifier.pickle')
    nb_classifier = pickle.load(f)
    f.close()
    print(accuracy(nb_classifier, test_feats))
    print(nb_classifier.classify(negfeat))

    for x in range(0, 50):
        print(nb_classifier.classify(negfeat))


