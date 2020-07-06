# import statments
import numpy
import re
import pandas as pd
import random

'''
Tokenize each the sentences, example
Input : "John likes to watch movies. Mary likes movies too"
Ouput : "John","likes","to","watch","movies","Mary","likes","movies","too"
'''


def tokenize(sentences):
    words = []
    dic = {}
    for sentence in sentences:
        w = word_extraction(sentence)
        for i in w:
            if i not in dic.keys():
                dic[i] = 0
            dic[i] = dic[i] + 1
            if dic[i] > 1:
                words.append(i)
    words = sorted(list(set(words)))
    return words


def word_extraction(sentence):
    ignore = "a the is".split()
    words = re.sub("^[0-9*#+]+$", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab));

    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1



