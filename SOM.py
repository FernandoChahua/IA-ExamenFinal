import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


class SOM:
    def __init__(self, dim, generations=14000, factor=0.5, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.dim = dim
        self.factor = factor
        self.generations = generations
        self.weights = np.random.randn(self.rows, self.cols, self.dim)
        self.map = np.empty(shape=(self.rows, self.cols), dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                self.map[i][j] = []

    def euc_dist(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def manhattan_dist(self, r1, c1, r2, c2):
        return np.abs(r1 - r2) + np.abs(c1 - c2)

    def most_common(self, lst, n):
        if len(lst) == 0:
            return 0

        counts = np.zeros(shape=n, dtype=np.int)

        for i in range(len(lst)):
            counts[lst[i]] += 1
        return np.argmax(counts)

    def min_nodo(self, dato):
        result = (0, 0)
        minDistance = 1.0e20
        for i in range(self.rows):
            for j in range(self.cols):
                ed = self.euc_dist(self.weights[i][j], dato)
                if ed < minDistance:
                    minDistance = ed
                    result = (i, j)
        return result

    def process(self, data):
        maxRange = self.rows + self.cols

        for s in range(self.generations):
            alpha = 1.0 - (s * 1.0) / self.generations
            actualAlpha = alpha * self.factor
            actualRange = int(alpha * maxRange)
            t = np.random.randint(len(data))
            (bmu_row, bmu_col) = self.min_nodo(data[t])
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.manhattan_dist(bmu_row, bmu_col, i, j) < actualRange:
                        self.weights[i][j] = self.weights[i][j] + actualAlpha * (data[t] - self.weights[i][j])

    def tagging(self, data, tag):
        for t in range(len(data)):
            (m_row, m_col) = self.min_nodo(data[t])
            self.map[m_row][m_col].append(tag[t])

    def visualization(self):
        label_weights = np.zeros(shape=(self.rows, self.cols), dtype=np.int)

        for i in range(self.rows):
            for j in range(self.cols):
                label_weights[i][j] = self.most_common(self.map[i][j], 20)

        plt.imshow(label_weights)
        plt.colorbar()
        plt.show()

    def group(self, dato):
        (g_row, g_col) = self.min_nodo(dato)
        return self.most_common(self.map[g_row][g_col], 40)
