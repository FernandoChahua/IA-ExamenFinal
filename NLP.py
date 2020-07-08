# import statments
import numpy as np
import re
import time
import pandas as pd
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


class NLPSimpleText:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        pos = "love good great nice like awesome happy well amaze best excite beautiful good cool enjoy cute funny fantastic perfect"
        neg = "headache sad cancer out miss lose damn sick down bad off sorry hate poor suck never stupid hard gone hurt worst"
        vocab = pos + " " + neg
        self.vocab = vocab.split()

    def word_extraction(self, sentence):
        sentence = sentence.lower()
        words = re.sub("[^\w]", " ", sentence).split()
        return words

    def lemmatize_words(self, word):
        lemmatize_word = [self.lemmatizer.lemmatize(word), self.lemmatizer.lemmatize(word, pos="v"),
                          self.lemmatizer.lemmatize(word, pos="a"), self.lemmatizer.lemmatize(word, pos="n"),
                          self.lemmatizer.lemmatize(word, pos="r")]
        return lemmatize_word

    def interpret(self, sentence):

        words = self.word_extraction(sentence)
        bag_vector = np.zeros(len(self.vocab))
        flag = True
        for w in words:
            lemmatize_words_w = self.lemmatize_words(w)
            for i, word in enumerate(self.vocab):
                if word in lemmatize_words_w:
                    bag_vector[i] += 1
                    flag = False
        # if flag:
        #    print(sentence)
        # print(bag_vector)
        return bag_vector, flag

    def generate_bow(self, all_sentences, tag, pos=0):
        inputs = []
        for i, sentence in enumerate(all_sentences):
            bag, flag = self.interpret(sentence)
            inputs.append(bag)
            if flag:
                tag[i + pos] = 0
        return inputs


