# import statments
import numpy
import re
import pandas as pd
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


def word_extraction(sentence):
    sentence = sentence.lower()
    words = re.sub("[^\w]", " ", sentence).split()
    return words


def interpret(sentence):
    vocab = "fuck nice like love amazing hate beautiful".split()
    words = word_extraction(sentence)
    bag_vector = numpy.zeros(len(vocab))
    for w in words:
        lemmatize_word = lemmatizer.lemmatize(w, pos="v")
        for i, word in enumerate(vocab):
            if word == lemmatize_word:
                bag_vector[i] += 1
    print(bag_vector)


def generate_bow(allsentences):
    for sentence in allsentences:
        interpret(sentence)


# Using readlines()
file1 = open('datasets/train.txt', 'r', encoding='utf8')
Lines = file1.readlines()

generate_bow(Lines)
