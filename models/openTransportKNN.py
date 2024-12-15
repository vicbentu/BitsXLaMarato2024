import pandas as pd
from sklearn.neighbors import NearestNeighbors
import math
import time
from functools import lru_cache

class OpenTransportKNN:
    def __init__(self):
        self.df = pd.read_csv('./modDatasets/OpenTransportMap_Modified.csv')

    @lru_cache(maxsize=8)
    def predict(self, x, y, threshold=0.003):
        ret = []
        lels = []

        data2D = self.df[['longitud', 'latitud']]

        knn = NearestNeighbors(n_neighbors=300, algorithm='auto').fit(data2D)

        # Find the k-nearest neighbors for all points in your array
        distances, indices = knn.kneighbors([[x, y]])

        for j, v in enumerate(distances[0]):
            if v <= threshold:
                lels.append([indices[0][j], v])

        calcul = [0, 0, 0, 0, 0, 0]
        for i, d in lels:
            aux = 1 - d / threshold
            e = self.df['functional'][i]
            if e == 'fifthClass':
                calcul[0] += aux
            if e == 'fourthClass':
                calcul[1] += aux
            if e == 'thirdClass':
                calcul[2] += aux
            if e == 'secondClass':
                calcul[3] += aux
            if e == 'firstClass':
                calcul[4] += aux
            if e == 'mainRoad':
                calcul[5] += aux
        ret.append(calcul)

        return ret