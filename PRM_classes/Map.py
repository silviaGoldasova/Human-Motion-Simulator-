import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.interpolate import splprep, splev
from PRM_classes.Obstacle import Obstacle
from PRM_classes import PRMController
from .pathwayGeneration import *

numSamples = 800

class Map:

    def __init__(self, filename="Environment/environment.txt"):
        self.allObs = self.readObstacles(filename)
        self.fig = plt.figure()
        plt.clf()

    def drawMap(self):
        #plt.figure()
        currentAxis = plt.gca()
        for ob in self.allObs:
            if ob.isWall:
                x = [item[0] for item in ob.allCords]
                y = [item[1] for item in ob.allCords]
                plt.scatter(x, y, c="black")
                plt.plot(x, y, c="black")
            else:
                currentAxis.add_patch(Rectangle(
                    (ob.bottomLeft[0], ob.bottomLeft[1]), ob.width, ob.height, alpha=0.4))

        self.fig.canvas.draw()
        plt.show(block=False)

    def readObstacles(self, filename):

        env = open(filename, "r")

        mapInfo = env.readline()
        size = list(map(int, mapInfo.split(",")))
        self.width = int(size[0])
        self.height = int(size[1])

        allObs = np.zeros(0)
        obst_sides = np.zeros((0, 4))
        shopping_area_lenCum = np.zeros(0)
        #shopping_areas_dict = {} # dict of tuples ([x,y], [dx,dy], length_hrany)
        attract_sides = []
        prev = 0

        for i, obstacle in enumerate(env):   # for each obstacle
            obs_corners = obstacle.strip().split(";")
            cornerOne = list(map(int, obs_corners[0].split(",")))
            cornerTwo = list(map(int, obs_corners[1].split(",")))
            obs = Obstacle(cornerOne, cornerTwo)
            #obs.printFullCords()
            allObs = np.append(allObs, obs)

            # side of the obstacle that need to be taken into account
            obst_sides_arr = list(obs_corners[2].split(","))
            attraction_sides_arr = list(obs_corners[3].split(","))

            for side in obst_sides_arr:
                if side == 'L':
                    obst_sides = np.append(obst_sides, np.array([obs.topLeft, obs.bottomLeft]).reshape(1, 4), axis=0)
                    # obst_sides = np.concatenate((obst_sides, np.array([obs.topLeft, obs.bottomLeft]).flatten()), axis=0)
                elif side == 'R':
                    obst_sides = np.append(obst_sides, np.array([obs.topRight, obs.bottomRight]).reshape(1, 4), axis=0)
                elif side == 'T':
                    obst_sides = np.append(obst_sides, np.array([obs.topLeft, obs.topRight]).reshape(1, 4), axis=0)
                elif side == 'B':
                    obst_sides = np.append(obst_sides, np.array([obs.bottomLeft, obs.bottomRight]).reshape(1, 4), axis=0)

            # side of the obstacle that can attract people
            edge_len = 0
            for side in attraction_sides_arr:
                if side == 'L':
                    edge_len = obs.topLeft[1] - obs.bottomLeft[1]     # length of the wall
                    shopping_area_lenCum = np.append(shopping_area_lenCum, prev+edge_len)
                    attract_sides.append((obs.bottomLeft, obs.topLeft, [0,edge_len], edge_len))   # #4=offset for corner collision evasion
                elif side == 'R':
                    edge_len = obs.topRight[1] - obs.bottomRight[1]
                    shopping_area_lenCum = np.append(shopping_area_lenCum, prev+edge_len)
                    attract_sides.append((obs.bottomRight, obs.topRight, [0,edge_len], edge_len))   #
                elif side == 'T':
                    edge_len = obs.topRight[0] - obs.topLeft[0]
                    shopping_area_lenCum = np.append(shopping_area_lenCum, prev+edge_len)
                    attract_sides.append((obs.topLeft, obs.topRight, [edge_len,0], edge_len))
                elif side == 'B':
                    edge_len = obs.bottomRight[0] - obs.bottomLeft[0]
                    shopping_area_lenCum = np.append(shopping_area_lenCum, prev+edge_len)
                    attract_sides.append((obs.bottomLeft, obs.bottomRight, [edge_len,0], edge_len))
                prev += edge_len
            #print(shopping_area_lenCum)

        self.obst_sides_arr = obst_sides
        self.attraction_sides_arr = attract_sides
        self.shoppingAreaLenCum = shopping_area_lenCum
        return allObs

    def drawWaypoint(self, point, color='green'):
        #print("point to draw ", point[0], point[1])
        plt.scatter(point[0], point[1], s=60, c=color)

    def readCheckpoints(self):
        # obstacle format: obstacle per line: BL_x,BL_y; TR_x,TRy
        env = open("checkpoints.txt", "r")
        checkpoints = {}

        for i, person in enumerate(env):
            person_checkpoints = person.strip().split(";")
            checkpoints[i] = np.array([list(map(int, position.split(","))) for position in person_checkpoints])

        return checkpoints

    def plotPath(self, pathPoints, color="blue"):
        c = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        plt.plot(pathPoints[:,0], pathPoints[:,1], c=c, linewidth=1.5)

    def smoothPath(self, pathPoints):
        if pathPoints.size <= 6:
            return pathPoints
        tck, u = splprep([pathPoints[:,0], pathPoints[:,1]], s=20) # k=spline degree,  1 <= k <= 5, default is 3
        new_points = splev(u, tck)
        smoothedPath = np.array([new_points[0], new_points[1]]).T
        return smoothedPath

    def getRandomPaths(self, numOfInterestPoints, gateWayPoints, n_groups):
        # init PRM module needed for path generation from sample points
        prm = PRMController.PRMController(numSamples, self)

        # generate points
        rndInterestPoints, closestCornerPoints = generateRandomInterestPoints(numOfInterestPoints, self, prm)

        #for el in rndInterestPoints:
        #    self.drawWaypoint(el, 'green')

        # add all points to the prm module so they are used to form pathways
        gateWayPoints = replace_if_interest_p_collision(prm, gateWayPoints, self.width, self.height)
        points = np.append(gateWayPoints.reshape(-1, 2), closestCornerPoints.reshape(-1, 2), axis=0)
        points = np.append(points, rndInterestPoints.reshape(-1, 2), axis=0)
        prm.addSamplePoints(points)  # corner points and interest points

        prm.runPRM()  # builds paths between random points + important points

        # distribute interest points
        interestCheckpoints, waypointsIdx = distributeInterestPoints(rndInterestPoints, n_groups.size, )

        # generate path from the corner points
        cornersPathOrder, origCornersPathOrder = generateCornersPathOrder(prm, self, closestCornerPoints, gateWayPoints)

        # waypoint sequences for each group
        # dict_waypoints, waypoint_counts, group_waypoint_idx, starting_points = define_waypoints_for_groups(n_groups, mapE)
        orderedWaypoints = orderWaypoints(rndInterestPoints, closestCornerPoints, waypointsIdx, origCornersPathOrder, interestCheckpoints, gateWayPoints)

        # generate a route passing through the waypoints
        routes, isInterestDict = map_PRM_init(prm, self, orderedWaypoints)
        writePathToFile(routes, isInterestDict)

        dict_waypoints, waypoint_counts, group_waypoint_idx, starting_points = get_paths_info(n_groups, routes)

        return dict_waypoints, waypoint_counts, group_waypoint_idx, starting_points, isInterestDict

