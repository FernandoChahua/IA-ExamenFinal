import sklearn
from sklearn.feature_extraction.text import * 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
word = "hated"
lemmatize_word = [lemmatizer.lemmatize(word), lemmatizer.lemmatize(word, pos="v"),
                          lemmatizer.lemmatize(word, pos="a"),lemmatizer.lemmatize(word, pos="n"),
                          lemmatizer.lemmatize(word, pos="r")]
print(lemmatize_word)

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


cars_for_sell = [line.replace("\n", "") for line in open("datasets/negative_tweets.txt",encoding='UTF8')]
common_words = get_top_n_words(cars_for_sell, 100)
for word, freq in common_words:
    print(word, freq)