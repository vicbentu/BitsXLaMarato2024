import pandas as pd
from sklearn.neighbors import NearestNeighbors
import math

def predict(puntsEntrada, threshold):
    df = pd.read_csv('./modDatasets/OpenTransportMap_Modified.csv')
    ret = []
    lels = [[] for _ in range(len(puntsEntrada))]
    counter = 0

    for _, elem in df.iterrows():
        ela = elem['latitud']
        elo = elem['longitud']
        for index, (longitud, latitud) in enumerate(puntsEntrada):
            dist = math.sqrt((ela - latitud)**2 + (elo - longitud)**2)
            if dist <= threshold:
                lels[index].append((elem['functional'], dist))
        counter += 1
        print(counter)
        
    print("HALFWAY")
    for point in lels:
        calcul = [0, 0, 0, 0, 0, 0]
        for e, d in point:
            aux = 1 - d / threshold
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


answ = predict([[2.009802, 41.39216], [1.191975, 41.11588]], 0.003)
print(answ)