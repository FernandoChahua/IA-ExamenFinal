import numpy as np
import pandas as pd
import math
import copy
import time
from matplotlib import pyplot as plt


# Metodos para hallar la funcion sigmoidea
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)





# Clase de BackPropagation con sus metodos
class BackProgation:
    def __init__(self, entradas, salidas_esperadas, epocas=24000, const_aprendizaje=0.1, neuronas_capa_entrada=4,
                 neuronas_capa_oculta=4, neuronas_capa_salida=1):
        np.random.seed(1)
        self.epocas = epocas
        self.const_aprendizaje = const_aprendizaje
        self.neuronas_capa_entrada = neuronas_capa_entrada
        self.neuronas_capa_oculta = neuronas_capa_oculta
        self.neuronas_capa_salida = neuronas_capa_salida
        self.salida = []
        self.entradas = entradas
        self.salidas_esperadas = salidas_esperadas
        self.pesos_capaOculta = np.random.uniform(size=(self.neuronas_capa_entrada, self.neuronas_capa_oculta))
        self.bias_capaOculta = np.random.uniform(size=(1, self.neuronas_capa_oculta))
        self.pesos_capaSalida = np.random.uniform(size=(self.neuronas_capa_oculta, self.neuronas_capa_salida))
        self.bias_capaSalida = np.random.uniform(size=(1, self.neuronas_capa_salida))
        self.errores = []

    # Muestra los datos iniciales de la red
    def mostar_datos_iniciales(self):
        print("Pesos iniciales de la capa oculta: \n", end='')
        print(*self.pesos_capaOculta)
        print("BIAS inicial de la capa oculta: \n", end='')
        print(*self.bias_capaOculta)
        print("Pesos iniciales de la capa de salida: \n", end='')
        print(*self.pesos_capaSalida)
        print("BIAS inicial de la capa de salida: \n", end='')
        print(*self.bias_capaSalida)

    # Entrena la red una cantidad de epocas
    def entrenar(self):
        for _ in range(self.epocas):
            # Forward Propagation - Avanza hacia adelante
            hidden_layer_activation = np.dot(self.entradas, self.pesos_capaOculta)
            hidden_layer_activation += self.bias_capaOculta
            hidden_layer_output = sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output, self.pesos_capaSalida)
            output_layer_activation += self.bias_capaSalida

            predicted_output = sigmoid(output_layer_activation)
            self.salida = predicted_output

            # Backpropagation - Regresa para encontrar error y variaciones
            error = self.salidas_esperadas - predicted_output
            self.errores.append(math.sqrt(np.square(error).mean()))
            d_predicted_output = error * sigmoid_derivative(predicted_output)

            error_hidden_layer = d_predicted_output.dot(self.pesos_capaSalida.T)
            d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

            # Actualizando Pesos y BIAS
            self.pesos_capaSalida += hidden_layer_output.T.dot(d_predicted_output) * self.const_aprendizaje
            self.bias_capaSalida += np.sum(d_predicted_output) * self.const_aprendizaje
            self.pesos_capaOculta += self.entradas.T.dot(d_hidden_layer) * self.const_aprendizaje
            self.bias_capaOculta += np.sum(d_hidden_layer) * self.const_aprendizaje

    # Muestra los datos finales y resultados obtenidos del entrenamiento
    def mostrar_datos_finales(self):
        print("Peso finales de la capa oculta: \n", end='')
        print(self.pesos_capaOculta)
        print("BIAS finales de la capa oculta: \n", end='')
        print(self.bias_capaOculta)
        print("Peso finales de la capa salida:  \n", end='')
        print(self.pesos_capaSalida)
        print("BIAS finales de la capa salida: \n", end='')
        print(self.bias_capaSalida)
        print("\nSalidas de la red neuronal luego de 10,000 epocas: \n", end='   ')
        print(self.salida)
        print("Errores cuadraticos medios de las epocas\n")
        print(self.errores)

    # Predice el tipo de sentimiento al que pertenece la oracion, entre Positivo - Negativo - Neutro
    def predecir(self, sentence):
        # Realiza el proceso de avance en la neurona ya entrenada para hallar el resultado
        hidden_layer_activation = np.dot(sentence, self.pesos_capaOculta)
        hidden_layer_activation += self.bias_capaOculta
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.pesos_capaSalida)
        output_layer_activation += self.bias_capaSalida

        predicted_output = sigmoid(output_layer_activation)

        # El resultado devuelto lo clasificamos para que devuelva si es de tipo Positivo - Negativo - Neutro
        menordist = 500000
        resultados = [0.0, 0.5, 1.0]
        salida = 0
        for i, item in enumerate(resultados):
            if abs(predicted_output - item) < menordist:
                menordist = abs(predicted_output - item)
                salida = i
        prediccion = "Neutral"
        if (salida == 1):
            prediccion = "Negativo"
        elif (salida == 2):
            prediccion = "Positivo"
        return prediccion

    # Devuelve el error medio del arreglo de errores generado en el BackPropagation
    def errormedio(self, listaError):
        suma = 0
        for item in listaError:
            suma = suma + pow(item, 2)
        suma = suma / len(listaError)
        return math.sqrt(suma)

    # Grafica el error cuadratico medio para comparar resultados
    def graficar_error(self, titulo="RED BACKPROPAGATION"):
        x = np.arange(0, self.epocas, 1)
        self.errores = np.array(self.errores)
        y = self.errores[x]
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(titulo)
        plt.show()
        print(self.errores[len(self.errores) - 1])
