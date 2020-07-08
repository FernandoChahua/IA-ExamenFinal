# import statments
import numpy as np
import re
import time
import pandas as pd
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import Self_Organizing_Map
from red_back_propagation import BackProgation


class NLPSimpleText:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        pos = "love good great nice like awesome happy well amaze best excite beautiful good cool enjoy cute funny fantastic perfect"
        neg = "headache sad cancer out miss lose damn sick down bad off sorry hat poor suck never stupid hard gone hurt worst"
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


nlp = NLPSimpleText()

negatives = open("datasets/negative_tweets.txt", 'r', encoding='utf8').readlines()
positives = open("datasets/positive_tweets.txt", 'r', encoding='utf8').readlines()
input_data = []
tag = []
for i in range(1000):
    input_data.append(negatives[i])
    tag.append(1)
    input_data.append(positives[i])
    tag.append(2)

data_input = np.array(nlp.generate_bow(input_data[0:1000], tag, 0))
data_test = np.array(nlp.generate_bow(input_data[1000:2000], tag, 1000))
# print(tag)

som = Self_Organizing_Map.SOM(40)
som.process(data_input)
som.tagging(data_input, tag)
som.visualization()
count = 0

input_data_back_propagation = data_test
output_data_back_propagation = []

for i in range(1000):
    if som.group(data_test[i]) == tag[i + 1000]:
        count = count + 1
    output_data_back_propagation.append([som.group(data_test[i]) / 2])
output_data_back_propagation = np.array(output_data_back_propagation)
print(output_data_back_propagation[0:10])
print(input_data_back_propagation[0:10])
print(len(output_data_back_propagation))
print(len(input_data_back_propagation))

back_propagation = BackProgation(input_data_back_propagation, output_data_back_propagation, neuronas_capa_entrada=40,
                                 neuronas_capa_salida=1, neuronas_capa_oculta=20)
back_propagation.entrenar()

print(count)
