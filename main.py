from NLP import NLPSimpleText
from SOM import SOM
from BackPropagation import BackProgation
import numpy as np

# Se instancia la clase de procesamiento de texto
nlp = NLPSimpleText()

# Leemos nuestro dataset en este caso tweets negativos y positivos
negatives = open("datasets/negative_tweets.txt", 'r', encoding='utf8').readlines()
positives = open("datasets/positive_tweets.txt", 'r', encoding='utf8').readlines()
input_data = []
tag = []
# Generamos la etiqueta de los datos manualmente
for i in range(1000):
    input_data.append(negatives[i])
    tag.append(1)
    input_data.append(positives[i])
    tag.append(2)

# Tenemos un dataset de 2000 por lo que entrenaremos con 1000 oraciones la red SOM
# Y los otros 1000 los etiquetará la red SOM
data_input = np.array(nlp.generate_bow(input_data[0:1000], tag, 0))
data_test = np.array(nlp.generate_bow(input_data[1000:2000], tag, 1000))
data_test_words = input_data[1000:2000]

print("Creando Red SOM")

# Instanciamos nuestra clase SOM con dimension 40 que sera la cantidad de inputs en total de nuestra red
som = SOM(40)

print("Procesando Red SOM.......")

# Procesamos los datos con los datos de entrada
som.process(data_input)
# Etiquetamos los nodos de la red som para obtener una respuesta al momento de
# que ejecutemos el agrupamiento
som.tagging(data_input, tag)
# Generamos el grafico de la red SOM
som.visualization()
count = 0

input_data_back_propagation = data_test
output_data_back_propagation = []

for i in range(1000):
    if som.group(data_test[i]) == tag[i + 1000]:
        count = count + 1
    output_data_back_propagation.append([som.group(data_test[i]) / 2])

output_data_back_propagation = np.array(output_data_back_propagation)

print("CREANDO RED NEURONAL 1")
back_propagation1 = BackProgation(input_data_back_propagation, output_data_back_propagation, neuronas_capa_entrada=40,
                                  neuronas_capa_salida=1, neuronas_capa_oculta=10, epocas=30000)
print("ENTRENANDO RED NEURONAL 1")
back_propagation1.entrenar()
back_propagation1.graficar_error("RED NEURONAL(CAPAS 10)")

print("CREANDO RED NEURONAL 2")
back_propagation2 = BackProgation(input_data_back_propagation, output_data_back_propagation, neuronas_capa_entrada=40,
                                  neuronas_capa_salida=1, neuronas_capa_oculta=20, epocas=30000)
print("ENTRENANDO RED NEURONAL 2")
back_propagation2.entrenar()
back_propagation2.graficar_error("RED NEURONAL(CAPAS 20)")

print("CREANDO RED NEURONAL 3")
back_propagation3 = BackProgation(input_data_back_propagation, output_data_back_propagation, neuronas_capa_entrada=40,
                                  neuronas_capa_salida=1, neuronas_capa_oculta=30, epocas=30000)
print("ENTRENANDO RED NEURONAL 3")
back_propagation3.entrenar()
back_propagation3.graficar_error("RED NEURONAL(CAPAS 30)")

print("CREANDO RED NEURONAL 4")
back_propagation4 = BackProgation(input_data_back_propagation, output_data_back_propagation, neuronas_capa_entrada=40,
                                  neuronas_capa_salida=1, neuronas_capa_oculta=40, epocas=30000)
print("ENTRENANDO RED NEURONAL 4")
back_propagation4.entrenar()
back_propagation4.graficar_error("RED NEURONAL(CAPAS 40)")

for i, word_interpret in enumerate(data_test):
    print("-------------------------------------------------------------------------")
    print("PRUEBA CON : ", data_test_words[i])
    print("RED NEURONAL 1(Capas 10):", back_propagation1.predecir([word_interpret]))
    print("RED NEURONAL 2(Capas 20):", back_propagation2.predecir([word_interpret]))
    print("RED NEURONAL 3(Capas 30):", back_propagation3.predecir([word_interpret]))
    print("RED NEURONAL 4(Capas 40):", back_propagation4.predecir([word_interpret]))
    print("-------------------------------------------------------------------------")
