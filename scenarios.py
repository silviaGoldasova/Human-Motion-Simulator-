import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import csv
import random
pi = math.pi
import json

def pointsInCircle(r, n=10):
    return [ (math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r) for x in range(0, n+1)]

def readRealDataset():
    file = open("testDb.csv")
    csvreader = csv.reader(file, delimiter=" ")
    header = next(csvreader)
    print(header)
    table = np.zeros((33897, 45*3+2))
    for i, row in enumerate(csvreader):
        row = np.array(row)
        row = row[0].strip().split(";")
        l = len(row)
        rowFloat = np.zeros(l)
        for j, el in enumerate(row):
            if el.replace('.','',1).replace('-','',1).isdigit():
                rowFloat[j] = float(el)
            else:
                rowFloat[j] = 0
                print(el)
        table[i, :] = rowFloat.reshape((1,-1))

    print(table.shape)

    file.close()

    t = table[:,1]

    return table

def displayData(table):
    #input np array (frames/time x persons)

    fig = plt.figure()

    for k in range(5):
        i=k+20
        idxX = 2 + i * 3
        idxY = 3 + i * 3
        idxZ = 4 + i * 3
        x = table[:, idxX]
        y = table[:, idxY]
        z = table[:, idxZ]

        # data for one person in x,y
        # filter the sections when no data available

        filtratedPoints = np.zeros((0, 3))
        for j in range(x.size):
            if x[j]==0 and y[j]==0 and z[j]==0:
                if filtratedPoints.size > 4: # at least 2 points
                    plt.plot(filtratedPoints[:,0], filtratedPoints[:,1])
                filtratedPoints = np.zeros((0, 3))
            else:
                filtratedPoints = np.append(filtratedPoints.reshape(-1,3), np.array([x[j],y[j],z[j]]).reshape(1,3), axis=0)

    fig.savefig("trajDataset3.png", dpi=300)
    plt.show()

def getRandomColor():
    color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color

if __name__ == '__main__':

    arr = pointsInCircle(6, 6)

    for i, el in enumerate(arr):
        id2 = i+3
        id2 = id2 % 6
        print(arr[i][0], ",", arr[i][1], ";", arr[id2][0], ",", arr[id2][1])

    f = open("Environment/entryShop.txt", "r")
    coords = np.zeros((2,2))

    coordsStart = f.readline()
    coords[0,:] = list(map(int, coordsStart.split(",")))
    coordsEnd = f.readline()
    coords[1, :] = list(map(int, coordsEnd.split(",")))

    print(coords)

    groupCounts = np.random.normal(1, 2, size=(5))
    groupCounts = np.array([int(abs(el)) for el in groupCounts])
    groupCounts = np.where(groupCounts == 0, 1, groupCounts)
    print(groupCounts)

    with open('Environment/configFile.txt') as f:
        data = f.read()
        js = json.loads(data)
        print(js)
    f.close()
