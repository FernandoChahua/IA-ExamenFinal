from NLP import NLPSimpleText
from SOM import SOM
from BackPropagation import BackProgation
import numpy as np

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

som = SOM(40)
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

print("Ingrese su oracion:\n")
while(True):
    sentence = input()
    print(back_propagation.predecir([nlp.interpret(sentence)[0]]))
    print('\n')
#back_propagation.mostrar_datos_finales()
#back_propagation.graficar_error()

print(count)
