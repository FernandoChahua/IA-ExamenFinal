# import statments
import numpy
import re
import pandas as pd
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import Self_Organizing_Map

lemmatizer = WordNetLemmatizer()


def word_extraction(sentence):
    sentence = sentence.lower()
    words = re.sub("[^\w]", " ", sentence).split()
    return words


def interpret(sentence):
    vocab = "fuck nice like love amaze hate beautiful".split()
    words = word_extraction(sentence)
    bag_vector = numpy.zeros(len(vocab))
    for w in words:
        lemmatize_word = lemmatizer.lemmatize(w, pos="v")
        for i, word in enumerate(vocab):
            if word == lemmatize_word:
                bag_vector[i] += 1

    print(bag_vector)
    return bag_vector


def generate_bow(allsentences):
    inputs = []
    for sentence in allsentences:
        inputs.append(interpret(sentence))
    return inputs


# Using readlines()
file1 = open('datasets/train.txt', 'r', encoding='utf8')
Lines = file1.readlines()

data_input = numpy.array(generate_bow(Lines[0:10]))
tag = numpy.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0])

som = Self_Organizing_Map.SOM(7)
som.process(data_input)
som.tagging(data_input,tag)
som.visualization()

for i in range(10):
    print(som.group(data_input[i]), tag[i])




