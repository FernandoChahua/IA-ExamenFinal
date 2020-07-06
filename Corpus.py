import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text

ruta = 'datasets'
archivo = 'positive_tweets.txt'
lectordetexto = PlaintextCorpusReader(ruta,archivo,encoding='utf8')
temp = lectordetexto.sents()
texto = Text(temp)
print(temp[0])

"""
print(texto.count('inteligencia'))
print(texto.index('inteligencia'))
print(texto.concordance('inteligencia'))
print("dispersion")
texto.dispersion_plot(['inteligencia','maquina'])
print("frecuencia")
Frecuencia = {'inteligencia':20,'maquina':11,'aprendizaje':5,'malo':1}
print(Frecuencia['inteligencia'])
print(Frecuencia.keys())
print(Frecuencia.values())
print(Frecuencia.items())
print(str(Frecuencia))
texto1 = ['hola','quetal','hola']
fdist = nltk.FreqDist(texto1)
print(fdist)
print(list(nltk.FreqDist.keys(fdist)))
print(list(nltk.FreqDist.values(fdist)))"""