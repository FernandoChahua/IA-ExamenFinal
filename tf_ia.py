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

lemmatizer = WordNetLemmatizer()


def word_extraction(sentence):
    sentence = sentence.lower()
    words = re.sub("[^\w]", " ", sentence).split()
    return words


def interpret(sentence):
    pos = "love good great nice like awesome happy well amaze best excite beautiful good cool enjoy cute funny fantastic perfect"
    neg = "headache sad cancer out miss lose damn sick down bad off sorry hat poor suck never stupid hard gone hurt worst"
    vocab = pos + " " + neg
    vocab = vocab.split()
    words = word_extraction(sentence)
    bag_vector = np.zeros(len(vocab))
    flag = True
    for w in words:
        lemmatize_word = []
        lemmatize_word.append(lemmatizer.lemmatize(w))
        lemmatize_word.append(lemmatizer.lemmatize(w, pos="v"))
        lemmatize_word.append(lemmatizer.lemmatize(w, pos="a"))
        lemmatize_word.append(lemmatizer.lemmatize(w, pos="n"))
        lemmatize_word.append(lemmatizer.lemmatize(w, pos="r"))
        for i, word in enumerate(vocab):
            if word in lemmatize_word:
                bag_vector[i] += 1
                flag = False
    if(flag):
        print(sentence)
    #print(bag_vector)
    return bag_vector,flag


def generate_bow(allsentences,tag,pos):
    inputs = []
    for i,sentence in enumerate(allsentences):
        bag,flag = interpret(sentence)
        inputs.append(bag)
        if(flag):
            tag[i+pos] = 0
    return inputs



negatives = open("datasets/negative_tweets.txt",'r',encoding='utf8').readlines()
positives = open("datasets/positive_tweets.txt", 'r',encoding='utf8').readlines()
input_data = [] 
tag = []
for i in range(1000):
    input_data.append(negatives[i])
    tag.append(1)
    input_data.append(positives[i])
    tag.append(2)


data_input = np.array(generate_bow(input_data[0:1000],tag,0))
data_test = np.array(generate_bow(input_data[1000:2000],tag,1000))
#print(tag)

som = Self_Organizing_Map.SOM(40)
som.process(data_input)
som.tagging(data_input, tag)
som.visualization()
count = 0
for i in range(1000):
    
    if som.group(data_test[i]) == tag[i+1000]:
        count= count + 1
    else:
        print(som.group(data_test[i]), tag[i+1000])
print(count)
