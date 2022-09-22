from collections import defaultdict
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse
from PRM_classes import Map

from .Dijkstra import Graph, dijkstra, to_array
r_min = 1.5

class PRMController:
    def __init__(self, numOfRandomCoordinates, mapE, k=5):
        self.numOfCoords = numOfRandomCoordinates
        self.coordsList = np.zeros((0,2))
        self.mapE = mapE
        self.k = k
        self.collisionFreePoints = np.zeros((0,2))
        self.interestCoordsList = np.zeros((0,2))
        #self.genCoords()

    def addSamplePoints(self, points):
        self.interestCoordsList = np.append(self.interestCoordsList, points.reshape(-1,2), axis=0)
        #self.checkIfCollisonFreeAdded(points)

    def runPRM(self, saveImage=True):
        #initialRandomSeed = 0
        #np.random.seed()
        self.graph = Graph()

        # Generate n random samples called milestones
        self.genCoords()

        # Check if milestones are collision free
        self.checkIfCollisonFree()

        # Link each milestone to k nearest neighbours. Retain collision free links as local paths.
        self.findNearestNeighbour()

        #if(saveImage):
        #    plt.savefig("{}_samples.png".format(self.numOfCoords))
        #plt.show()

    def addPoints(self, pointsArr):
        pointsArr = pointsArr.reshape(-1,2)
        self.interestCoordsList = np.append(self.interestCoordsList.reshape(-1,2), pointsArr, axis=0)

        # check if collision free
        for point in pointsArr:
            #print("adding ", point)
            collision = self.checkPointCollision(point)

            #isUnique = True
            isUnique = not (np.equal(point, self.collisionFreePoints).all(axis=1).any())   # any => true if match
            hasEdges = False
            if not isUnique:
                id = self.findNodeIndex(point)
                hasEdges = (str(id) in self.graph.edges.keys())
                #print("hasEdges", hasEdges, id, np.where((self.collisionFreePoints == point.reshape(-1,2)).all(axis=1))[0], self.graph.edges.keys())
            if (not collision and not hasEdges):
                if isUnique:
                    self.collisionFreePoints = np.append(self.collisionFreePoints, point.reshape(1, 2), axis=0)
                #print("adding a point ", point, isUnique, hasEdges)

                # knn
                distances, indices = self.knn.kneighbors(point.reshape(1,-1))

                # Ignoring nearest neighbour - nearest neighbour is the point itself
                for j, neighbour in enumerate(self.collisionFreePoints[indices[0,:]]):
                    if np.array_equal(point, neighbour):
                        continue
                    start_line = point
                    end_line = neighbour
                    #print("new edge: ", point.reshape(1, 2), neighbour.reshape(1, 2))

                    # if no collision of the line with the obstacles
                    if (not self.checkLineCollision(start_line, end_line)):
                        self.collisionFreePaths = np.concatenate(
                            (self.collisionFreePaths, point.reshape(1, 2), neighbour.reshape(1, 2)), axis=0
                        )
                        a = str(self.collisionFreePoints.shape[0]-1)  # a = str(self.findNodeIndex(point)) => point is at index i      # lastly added elemenet -> last index => get num of rows
                        b = str(indices[0,j])  # b = str(self.findNodeIndex(neighbour))

                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[0,j])
                        self.graph.add_node(b)
                        self.graph.add_edge(b, a, distances[0,j])
                        x = [point[0], neighbour[0]]
                        y = [point[1], neighbour[1]]

                        #print("add points arr: ", x, y)
                        #self.mapE.plotPath(x, y, 'orange')
                        #plt.plot(x, y, c='orange');
                        #print("edges added: ", start_line, end_line)
            #else:
                #if not isUnique:
                #    id = self.findNodeIndex(point)
                #    print("not added to interest points: ", point, "because isUnique | collision: ", isUnique, collision, "idx: ", self.findNodeIndex(point), ", neighbours: ", self.graph.edges[id])

    # Search for shortest path from start to end node - Using Dijksta's shortest path alg
    def shortestPath(self, start, destination):
        self.addPoints(np.array(start))
        self.addPoints(np.array(destination))

        self.startNode = str(self.findNodeIndex(start))
        self.endNode = str(self.findNodeIndex(destination))

        if self.startNode == -1 or self.endNode == -1:
            print("nodes not found: ", self.startNode, self.endNode)

        dist, prev = dijkstra(self.graph, self.startNode)
        pathToEnd = to_array(prev, self.endNode)  # indices

        if (len(pathToEnd) < 1):
            return None

        points = [(self.findPointsFromNode(i)) for i in pathToEnd]
        pathPoints = np.array([(int(item[0]), int(item[1])) for item in points])

        return pathPoints, dist

    def genAdditCoords(self):
        map_width, map_height = self.mapE.width, self.mapE.height

        l = int(0.2*self.numOfCoords)
        coordsListX = np.random.randint(map_width, size=l)
        coordsListY = np.random.randint(map_height, size=l)
        newPts = np.array([coordsListX, coordsListY]).T
        return newPts

    def genCoords(self):
        map_width, map_height = self.mapE.width, self.mapE.height

        if self.coordsList.size < 1:
            coordsListX = np.random.randint(map_width, size=self.numOfCoords)
            coordsListY = np.random.randint(map_height, size=self.numOfCoords)
            self.coordsList = np.array([coordsListX, coordsListY]).T
        else:
            l = int(0.5*self.numOfCoords)
            coordsListX = np.random.randint(map_width, size=l)
            coordsListY = np.random.randint(map_height, size=l)
            newPts = np.array([coordsListX, coordsListY]).T
            self.coordsList = np.append(self.coordsList.reshape(-1,2), newPts, axis=0)

    def checkIfCollisonFreeAdded(self, points):
        collision = False
        collisionFreeTmp = np.zeros((0, 2))
        for point in points:
            collision = self.checkPointCollision(point)
            if (not collision):
                collisionFreeTmp = np.append(collisionFreeTmp.reshape(-1, 2), point, axis=0)

        collisionFreeTmp = np.unique(collisionFreeTmp, axis=0)
        self.collisionFreePoints = np.append(self.collisionFreePoints.reshape(-1,2), collisionFreeTmp, axis=0)

    def checkIfCollisonFree(self):
        self.coordsList = np.unique(self.coordsList, axis=0)
        self.interestCoordsList = np.unique(self.interestCoordsList, axis=0)

        collision = False
        self.collisionFreePoints = np.array([])
        for point in self.coordsList:
            collision = self.checkPointCollision(point, largerBorders=True)
            if(not collision):
                if(self.collisionFreePoints.size == 0):
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point])

        for point in self.interestCoordsList:
            collision = self.checkPointCollision(point)
            if(not collision):
                if(self.collisionFreePoints.size == 0):
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point])

        self.collisionFreePoints = np.unique(self.collisionFreePoints, axis=0)
        #self.plotPoints(self.collisionFreePoints)

    def findNearestNeighbour(self):
        X = self.collisionFreePoints
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(X)
        distances, indices = self.knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 2), int)

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                #if(not self.checkPointCollision(start_line) and not self.checkPointCollision(end_line)):

                # if no collision of the line with the obstacles
                if(not self.checkLineCollision(start_line, end_line)):
                    self.collisionFreePaths = np.concatenate(
                        (self.collisionFreePaths, p.reshape(1, 2), neighbour.reshape(1, 2)),axis=0
                    )
                    a = str(i)      #a = str(self.findNodeIndex(p)) => p is at index i
                    b = str(indices[i][j+1]) # b = str(self.findNodeIndex(neighbour))

                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j+1])
                    x = [p[0], neighbour[0]]
                    y = [p[1], neighbour[1]]
                    #plt.plot(x, y)

    def findNearestNeighbourForPoint(self, pt):
        ptId = str(self.findNodeIndex(pt))
        edges = self.graph.edges[ptId]
        print("edges from ", pt, " lead to ", edges)

    def checkLineCollision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.mapE.allObs:
            if obs.isWall:
                uniqueCords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(uniqueCords)
                if(line.intersection(wall)):
                    collision = True
            else:
                obstacleShape = shapely.geometry.Polygon(obs.allCords)
                collision = line.intersects(obstacleShape)
            if(collision):
                return True
        return False

    def findNodeIndex(self, p):
        #print(p.shape, p, self.collisionFreePoints.shape, np.where((self.collisionFreePoints == p).all(axis=1))[0])
        if np.where((self.collisionFreePoints == p.reshape(-1,2)).all(axis=1))[0].size >= 1:
            return np.where((self.collisionFreePoints == p.reshape(-1,2)).all(axis=1))[0][0]      # first elem = array of such elements/indices, than select first index
        else: # node not found
            return -1

    def findPointsFromNode(self, n):
        return self.collisionFreePoints[int(n)]

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y, c="black", s=1)

    def checkCollision(self, obs, point, largerBorders=False):
        p_x = point[0]
        p_y = point[1]

        r = 0
        if largerBorders:
            r = r_min

        if(obs.bottomLeft[0] - r <= p_x <= obs.bottomRight[0] + r and obs.bottomLeft[1] -r <= p_y <= obs.topLeft[1] + r):
            return True
        else:
            return False

    def checkPointCollision(self, point, largerBorders=False):
        for obs in self.mapE.allObs:
            collision = self.checkCollision(obs, point, largerBorders)
            if(collision):
                return True
        return False

    def printPathInfo(self, pathToEnd, dist):
        pointsToEnd = [str(self.findPointsFromNode(path)) for path in pathToEnd]
        print("****Output****")

        print("The quickest path from {} to {} is: \n {} \n with a distance of {}".format(
            self.collisionFreePoints[int(self.startNode)],
            self.collisionFreePoints[int(self.endNode)],
            " \n ".join(pointsToEnd),
            str(dist[self.endNode])
        ))

