#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


# In[87]:


class SOM():
    def __init__(self, dimension):
        self.rows = 15
        self.cols = 15
        self.dimension = dimension
        self.factor = 0.5
        self.iter = 1000
        self.pesos = np.random.randn(self.rows, self.cols, self.dimension)
        self.mapa = np.empty(shape=(self.rows, self.cols), dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                self.mapa[i][j] = []

    def euc_dist(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def manhattan_dist(self, r1, c1, r2, c2):
        return np.abs(r1 - r2) + np.abs(c1 - c2)

    def most_common(self, lst, n):
        if len(lst) == 0: return -1

        counts = np.zeros(shape=n, dtype=np.int)

        for i in range(len(lst)):
            counts[lst[i]] += 1
        return np.argmax(counts)

    def minimoNodo(self, dato):
        result = (0, 0)
        distanciaMinima = 1.0e20
        for i in range(self.rows):
            for j in range(self.cols):
                ed = self.euc_dist(self.pesos[i][j], dato)
                if ed < distanciaMinima:
                    distanciaMinima = ed
                    result = (i, j)
        return result

    def process(self, data):
        rangoMax = self.rows + self.cols

        for s in range(self.iter):
            alfa = 1.0 - (s * 1.0) / self.iter
            alfaActual = alfa * self.factor
            rangoActual = (int)(alfa * rangoMax)
            t = np.random.randint(len(data))
            (bmu_row, bmu_col) = self.minimoNodo(data[t])
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.manhattan_dist(bmu_row, bmu_col, i, j) < rangoActual:
                        self.pesos[i][j] = self.pesos[i][j] + alfaActual * (data[t] - self.pesos[i][j])

    def tagging(self, data, tag):
        for t in range(len(data)):
            (m_row, m_col) = self.minimoNodo(data[t])
            self.mapa[m_row][m_col].append(tag[t])

    def visualization(self):
        label_pesos = np.zeros(shape=(self.rows, self.cols), dtype=np.int)

        for i in range(self.rows):
            for j in range(self.cols):
                label_pesos[i][j] = self.most_common(self.mapa[i][j], 20)

        plt.imshow(label_pesos)
        plt.colorbar()
        plt.show()

    def group(self, dato):
        (g_row, g_col) = self.minimoNodo(dato)
        return self.most_common(self.mapa[g_row][g_col], 40)

