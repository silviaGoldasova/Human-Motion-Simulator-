import numpy as np
import random
from bisect import bisect_left
import matplotlib.pyplot as plt
import aux_functions

offset = np.array([[1,1], [1,-1], [-1,-1], [-1,1]])
offsetSideways = np.array([[0,1], [1,0], [0,-1], [-1,0]])
offsetAll = np.array([[1,1], [1,-1], [-1,-1], [-1,1], [0,1], [1,0], [0,-1], [-1,0]])
meanWaypoints = 6
sdWaypoints = 4

coefDist = 2

# Get effective order of the randomly generated interest points
def generateCornersPathOrder(prm, mapE, cornerPoints, gateWayPoints):

    # append also the starting point
    cornerPoints = np.append(cornerPoints, gateWayPoints[0].reshape((-1, 2)), axis=0)

    # get unique set of corner points without collisions
    cornerPointsU, unique_idx, inverse_idx = np.unique(cornerPoints.reshape(-1, 2), axis=0, return_index=True, return_inverse=True)
    cornerPointsS = replace_if_collision(prm, cornerPointsU, mapE.width, mapE.height)

    idxStartingPoint = inverse_idx[cornerPoints.shape[0] - 1]
    l = cornerPointsS.shape[0]

    # get distances between corner points
    pairsDistances = np.zeros((l, l)) + np.inf
    for j in range(l - 1):
        for i in range(j + 1, l):
            if np.array_equal(cornerPointsS[j, :], [-1, -1]) or np.array_equal(cornerPointsS[i, :], [-1, -1]):
                continue
            pair = np.array([cornerPointsS[j, :], cornerPointsS[i, :]])
            pair_dist = pair_shortestPath_dist(mapE, prm, pair)
            pairsDistances[j, i] = pairsDistances[i, j] = pair_dist

    routeCorners = np.zeros((l))
    unvisited = np.zeros((l)) + True
    prevNode = idxStartingPoint  # starting node is the first one
    routeCorners[0] = idxStartingPoint

    # first point = start, then always get the closest point from the unvisited points
    for i in range(l - 1):
        distsFromNode = pairsDistances[prevNode, :]
        distsFromNode[unvisited == False] = np.inf
        min_idx = np.argmin(distsFromNode)
        routeCorners[i + 1] = min_idx
        unvisited[min_idx] = False

    # get a path of points
    cornersPointsPath = np.zeros((l, 2))
    origArr = np.zeros((l, 2))
    for i in range(l):
        cornersPointsPath[i, :] = cornerPointsS[int(routeCorners[i]), :]
        origArr[i,:] = cornerPoints[unique_idx[int(routeCorners[i])],:]

    #print("origArr", origArr)

    return cornersPointsPath, origArr

# returns cost of the shortest path between two locations
def pair_shortestPath_dist(mapE, prm, points_pair):
    # waypoints is an array (n_pairs x (start_path,end_path) x 2_coords)

    curr = points_pair[0,:]
    waypoint = points_pair[1,:]
    mapE.drawWaypoint(curr)

    pathPoints, dist = prm.shortestPath(curr, waypoint)
    while pathPoints.size < 3:
        print("path not generated for ", curr,  ", ", waypoint)

        prm.addPoints(waypoint)
        #id = str(prm.findNodeIndex(waypoint))
        #print("idx: ", prm.findNodeIndex(waypoint), ", neighbours: ", prm.graph.edges[id])
        #prm.runPRM()

        newSampled = prm.genAdditCoords()
        prm.addPoints(newSampled)

        pathPoints, dist = prm.shortestPath(curr, waypoint)

    final_node_idx = str(prm.findNodeIndex(waypoint))   # get node idx in the graph
    pt_pair_distance = dist[final_node_idx]

    #mapE.plotPath(pathPoints, 'blue')
    #print("built route between ", curr, " and ", waypoint, ", cost=", pt_pair_distance)

    #plt.show()
    #plt.clf()

    return pt_pair_distance


# colors=['orange', 'green', 'blue', 'purple', 'red']
def map_PRM_init(prm, mapE, waypoints):

    routes = {}
    isPointOfInterest = {}
    for i, person_idx in enumerate(waypoints):  # person_idx are dict keys
        smoothedPath = np.zeros((0, 2))
        interests = np.zeros((0), dtype=int)
        curr = waypoints[person_idx][0]  # start of the person's route
        for waypoint in waypoints[person_idx][1:]:
            if np.array_equal(curr,waypoint):
                continue

            pathPoints, dist = prm.shortestPath(curr.reshape(-1,2), waypoint.reshape(-1,2))
            while pathPoints.size < 3:
                print("path not generated: ", curr, waypoint)

                prm.addPoints(waypoint)
                newSampled = prm.genAdditCoords()
                prm.addPoints(newSampled)

                pathPoints, dist = prm.shortestPath(curr, waypoint)

            pathPointsU, idx = np.unique(pathPoints, axis=0, return_index=True)
            pathPoints = np.array([pathPoints[index] for index in sorted(idx)])
            smoothedPathAdd = mapE.smoothPath(pathPoints)   # returns arr (numPoints, 2)
            smoothedPathAdd = checkPath(smoothedPathAdd, mapE, prm)
            mapE.plotPath(smoothedPathAdd)

            # actual interest points are only at the beginning and at the end
            interestsN = np.zeros(smoothedPathAdd.shape[0])
            interestsN[0] = interestsN[-1] = True
            interests = np.concatenate((interests, interestsN))

            smoothedPath = np.concatenate((smoothedPath, smoothedPathAdd))
            curr = waypoint

        routes[person_idx] = smoothedPath.T
        isPointOfInterest[person_idx] = interests

        pointsOfInt = [bool(el) for el in interests]
        for el in smoothedPath[pointsOfInt]:
            mapE.drawWaypoint(el, 'pink')

    plt.show(block=False)

    return routes, isPointOfInterest


def checkPath(smoothedPath, mapE, prm):
    for i, el in enumerate(smoothedPath):
        for obs_side in mapE.obst_sides_arr:
            nearestPoint = aux_functions.getNearestPointOnLine(obs_side[:2], obs_side[2:4], el)
            dist = np.linalg.norm(el - nearestPoint)
            if dist < 1.5:
                newPt = smoothedPath[i] + (np.divide(el - nearestPoint, dist)) * coefDist
                if(0 <= newPt[0] <= mapE.width and 0 <= newPt[1] <= mapE.height and not prm.checkPointCollision(newPt)):
                    print("too close ", nearestPoint, el, newPt)
                    smoothedPath[i] = newPt
    return smoothedPath

def distributeInterestPoints(interestPoints, numPeople):

    numOfIntPts = interestPoints.shape[0]
    waypoints = {}
    waypointsIdx={}
    for i in range(numPeople):
        waypointsCount = int(abs(np.random.normal(meanWaypoints, sdWaypoints)))
        rndPt = np.random.randint(numOfIntPts, size=(waypointsCount))
        waypointsIdx[i] = rndPt
        waypoints[i] = interestPoints[rndPt, :]

    return waypoints, waypointsIdx

def generateRandomInterestPoints(numOfPoints, mapE, prm):
    cornerPoints = np.zeros((0,2))
    totalLen = max(mapE.shoppingAreaLenCum)
    randomPts = np.zeros((0,2))
    point = None

    for j in range(numOfPoints):
        rndLen = random.randint(0, totalLen)
        idx = bisect_left(mapE.shoppingAreaLenCum, rndLen)  # find insertion position
        edge = mapE.attraction_sides_arr[idx]
        coef = random.random()
        point = edge[0] + np.array(edge[2]) * coef  # beggining of the edge + edge length * direction vector [dx,dy] * random number from (0,1)
        point = np.array([int(point[0]), int(point[1])])
        randomPts = np.append(randomPts, point.reshape(1,2), axis=0)

        closestCornerPtIdx = np.argmin([np.linalg.norm(edge[0]-point), np.linalg.norm(edge[1]-point)])
        cornerPoints = np.append(cornerPoints, (edge[closestCornerPtIdx]).reshape(1,2), axis=0)

    # get
    randomPts = replace_if_interest_p_collision(prm, randomPts, mapE.width, mapE.height)

    # remove where [-1,-1]
    notGeneratedIdx = np.where((randomPts[:, 0] == -1) & (randomPts[:, 1] == -1))[0]
    mask = np.ones(randomPts.shape[0], dtype=bool)
    mask[notGeneratedIdx] = False

    return randomPts[mask,:], cornerPoints[mask,:]

def orderWaypoints(rndInterestPoints, closestCornerPoints, waypointsIdx, cornersPathOrder, dict_interestCheckpoints, gateWayPoints):

    waypoints_ordered={}

    for i, person_idx in enumerate(dict_interestCheckpoints):
        id = waypointsIdx[i]
        orderedWaypoints = np.zeros((0,2))

        orderedWaypoints = np.append(orderedWaypoints, gateWayPoints[0])
        for j,cornerPoint in enumerate(cornersPathOrder):
            selectedPoints = np.where((closestCornerPoints[id,0] == cornerPoint[0]) & (closestCornerPoints[id,1]==cornerPoint[1]))[0]
            if selectedPoints.size > 0:
                orderedWaypoints = np.append(orderedWaypoints.reshape(-1,2), rndInterestPoints[id,:][selectedPoints,:], axis=0)

        orderedWaypoints = np.append(orderedWaypoints.reshape(-1,2), gateWayPoints[1].reshape(-1,2), axis=0)
        waypoints_ordered[i] = orderedWaypoints

    return waypoints_ordered

def get_paths_info(n_groups, routes):
    # starting_points{i} contains the starting point of group i
    starting_points = {}
    waypoint_counts = {}  # Number of waypoints
    groups_waypoints_idx = {}  # auxiliary index

    for i in range(n_groups.shape[0]):
        aux, waypoint_counts[i] = routes[i].shape  # number of waypoints of group i
        groups_waypoints_idx[i] = np.zeros((n_groups[i], 1), dtype=int)  # current waypoint for group i
        starting_points[i] = routes[i][:, 0]  # note initial point

    return routes, waypoint_counts, groups_waypoints_idx, starting_points

def replace_if_interest_p_collision(prm, pt_arr, map_w, map_h):
    # check if nodes exists
    success = False
    newArr = np.empty_like(pt_arr)

    for i, pt in enumerate(pt_arr):
        success = False
        if prm.findNodeIndex(pt) == -1:
            for dir in offsetSideways:
                p = pt+dir
                collision = prm.checkPointCollision(p)
                if (not collision and 0<p[0] and p[0]<map_w and 0<p[1] and p[1]<map_h):
                    #prm.addPoints(p)
                    newArr[i, :] = p
                    success = True
                    break
            if not success:
                newArr[i, :] = np.array([-1,-1])
        else:
            newArr[i,:] = pt_arr[i,:]
    return newArr

def replace_if_collision(prm, pt_arr, map_w, map_h):
    # check if nodes exists
    success = False
    newArr = np.empty_like(pt_arr)

    for i, pt in enumerate(pt_arr):
        success = False
        if prm.findNodeIndex(pt) == -1:
            for dir in offset:
                p = pt+dir
                collision = prm.checkPointCollision(p)
                if (not collision and 0<p[0] and p[0]<map_w and 0<p[1] and p[1]<map_h):
                    prm.addPoints(p)
                    newArr[i, :] = p
                    success = True
                    break
            if not success:
                newArr[i, :] = np.aray([-1,-1])
        else:
            newArr[i,:] = pt_arr[i,:]
    return newArr


def replace_if_collision_smoothing(pt, map_w, map_h, prm):
    success = False
    for coef in range(1, 5):
        coef_d = coef/25
        for dir in offsetAll:
            p = pt + coef_d*dir
            collision = prm.checkPointCollision(p, largerBorders=True)
            if (not collision and 0<p[0] and p[0]<map_w and 0<p[1] and p[1]<map_h):
                prm.addPoints(p)
                success = True
                break
    if not success:
        return pt
    return p

def correct_collision_points(arrUpdated, prm):
    noncollisionArr = np.zeros((0,2))
    for i, point in enumerate(arrUpdated):
        collision = prm.checkPointCollision(point, largerBorders=True)
        if (not collision):
            noncollisionArr = np.append(noncollisionArr, point.reshape(-1,2), axis=0)
        else:
            print("collision")
            newPt = replace_if_collision_smoothing(point, prm.mapE.width, prm.mapE.height, prm)
            print(newPt, " instead of ", point)
            noncollisionArr = np.append(noncollisionArr, newPt.reshape(-1, 2), axis=0)
            #p = arrOrig[i,:] + (arrOrig[i,:]-point)
            #noncollisionArr = np.append(noncollisionArr, p.reshape(-1,2), axis=0)
    return noncollisionArr

def writePathToFile(routes, interestsD):
    f = open("convert.txt", "w")
    f.close()
    f = open("interests.txt", "w")
    f.close()

    f = open("convert.txt", "a")
    for i in routes:
        line = ""
        for point in routes[i].T:
            string = "{},{};".format(point[0], point[1])
            line = line + string
        line = line[:-1] + '\n'
        f.write(line)

    f.close()

    f = open("interests.txt", "a")
    for i, key in enumerate(interestsD):
        f.write(','.join(str(int(char)) for char in interestsD[i]))
        f.write('\n')
    f.close()

def writeInterestsToFile(interestsD):
    f = open("interests.txt", "w")
    f.close()

    f = open("interests.txt", "a")
    for i, key in enumerate(interestsD):
        s = str(interestsD[i])
        s = s + '\n'
        f.write(s)
    f.close()

def readInterestsFromFile():
    interestsD = {}
    file = open("interests.txt", "r")
    for i, line in enumerate(file):
        line = line[1:-2]
        interests = np.array(list(map(int, line.split(" "))))
        interestsD[i] = interests
    file.close()


def readPathFromFile(pathsFile, interestsFile=None):
    routes={}
    interestsD = {}

    f = open(pathsFile, "r")
    for i, line in enumerate(f):
        checkpoints = line.strip().split(";")
        routes[i] = np.array([list(map(float, position.split(","))) for position in checkpoints]).T

    # generate own all file expressing that there are no points of interest
    if interestsFile == None:
        interestsD = routes.copy()
        for i in interestsD:
            interestsD[i] = np.zeros(routes[i].shape[1])
            interestsD[i][-1] = 1
        return routes, interestsD

    file = open(interestsFile, "r")
    for i, line in enumerate(file):
        interests = list(map(int, line.split(",")))
        interestsD[i] = np.array(interests)
    file.close()

    return routes, interestsD