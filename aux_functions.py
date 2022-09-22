import numpy as np
from PRM_classes import PRMController, Obstacle, Map
import matplotlib.pyplot as plt
import random
from agent import Agent, AgentGroup
from PRM_classes import pathwayGeneration
from fsm import States
import json

numSamples = 800

def readConfig():
    with open('Environment/configFile.txt') as f:
        data = f.read()
        js = json.loads(data)
        print(js)
    f.close()
    return js['mapFile'], js['routesFile'], js['startingPosFile'], js['interestsFile']

def map_def():
    walls = np.loadtxt("walls.txt", dtype=float, comments='#')
    num_walls, aux = walls.shape
    print(walls)

    return num_walls, walls

def generateGroupDistribution(numOfGroups):
    groupCounts = np.random.normal(1, 2, size=(numOfGroups))
    groupCounts = np.array([int(abs(el)) for el in groupCounts])
    groupCounts = np.where(groupCounts == 0, 1, groupCounts)
    return groupCounts

def define_waypoints_for_groups(n_groups, mapE):
    # starting_points{i} contains the starting point of group i
    starting_points = {}
    dict_waypoints = {}
    waypoint_counts = {}  # Number of waypoints
    groups_waypoints_idx = {}  # auxiliary index

    # set of arrays of waypoints for each person
    env = open("checkpoints.txt", "r")
    dict_waypoints = {}
    for i, person in enumerate(env):
        person_checkpoints = person.strip().split(";")
        dict_waypoints[i] = np.array([list(map(int, position.split(","))) for position in person_checkpoints])

    # map of person's routes = smoothed paths going through the waypoints

    prm = PRMController(numSamples, mapE)
    prm.runPRM()
    routes = map_PRM_init(mapE, dict_waypoints)

    for i in range(len(n_groups)):
        aux, waypoint_counts[i] = routes[i].shape  # number of waypoints of group i
        groups_waypoints_idx[i] = np.zeros((n_groups[i], 1), dtype=int)  # current waypoint for group i
        starting_points[i] = routes[i][:, 0]  # note initial point

    return routes, waypoint_counts, groups_waypoints_idx, starting_points

def getNearestPointOnLine(lineStart, lineEnd, point):
    #print(lineStart, lineEnd, personPos)
    x1, y1 = lineStart
    x2, y2 = lineEnd
    xP, yP = point

    div = ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if div != 0:
        t = ((xP - x1) * (x2 - x1) + (yP - y1) * (y2 - y1)) / div
    else:
        t=0
    t_star = min(max(0, t), 1)
    nearestPoint = lineStart + t_star * (lineEnd - lineStart)

    return nearestPoint

def parameters_load():

    # SFM Parameters
    tau = 0.5
    A = 2000
    B = 0.08
    Aw = 2000
    Bw = 0.08
    k1 = 1.2 * 10 ** 5
    k2 = 2.4 * 10 ** 5

    # HSFM Parameters
    #kd = 500    # a damping coefficient on the speed along the same direction
    kd = 100
    ko = 1.5      # a gain modulating the force acting on the direction orthogonal to the pedestrian's heading.
    k1g = 200   # forward group cohesion force strength
    k2g = 200   # sideward group cohesion force strength
    d_o = 0.5   # sideward maximum distance from the center of mass
    d_f = 1     # forward maximum distance from the center of mass
    alpha = 3

    waypoint_reached_coef = 1.5
    am = 1      # "am" represents the amplitude of the starting zone. Fix the starting points at least at "am" meters from the walls

    return tau, A, B, Aw, Bw, k1, k2, kd, ko, k1g, k2g, d_o, d_f, alpha, waypoint_reached_coef, am

def initialization(n_groups, N, rm, rM, mm, mM, v0m, v0M, s, am, mapE):
    # return
    # r     Radius (N,1) np array
    # m     Mass (N,1) np array
    # J     Inertia (N,1) np array
    # v0    Desired speed (scalar)
    # v     Initial zeroed speeds (N,2) np array
    # th    Initial orientation (scalar)
    # omg   Initial angular velocity (scalar)
    # group_membership  Group_membership (N,1) np array
    # X0    Initial conditions for derivative variables (N, 6) np array
    #                                                   (N x (starting posX, starting posY, initial head orientation theta, size of the initial speed (0), 0, initial angular velocity))
    # p     position vectos (N, 6) np array
    #                       (N x (initial posX, initial posY, initial velocity x, initial velocity y, radius, mass))

    # Map loading

    num_walls, obstacle_sides_arr = mapE.obst_sides_arr.shape[0], mapE.obst_sides_arr

    v0 = v0m + (v0M - v0m) * np.random.rand(N, 1)  # random desired speed
    v = 0 * np.ones((N, 2))  # initial speed
    th=0
    omg = 0  # initial angular velocity

    r = np.empty((N, 1), dtype=float)
    m = np.empty((N, 1), dtype=float)
    group_membership = np.empty((N, 1), dtype=int)
    for i in range(len(n_groups)):  # random radii and masses
        # random radii
        r[sum(n_groups[0: i + 1]) - n_groups[i]: sum(n_groups[0:i + 1])] = np.sort(
            rm + (rM - rm) * np.random.rand(n_groups[i], 1))
        # random masses
        m[sum(n_groups[0: i + 1]) - n_groups[i]: sum(n_groups[0:i + 1])] = np.sort(
            mm + (mM - mm) * np.random.rand(n_groups[i], 1))
        # aux variable
        group_membership[sum(n_groups[0: i + 1]) - n_groups[i]: sum(n_groups[0:i + 1])] = int(i)

    J = 0.5 * r ** 2  # Inertia

    i = 0
    p = {}
    X0 = []

    # for each agent
    while i < N:
        gr = int(group_membership[i])

        # generate a random starting point near the defined starting point
        pos = [s[gr][0] - am + 2 * am * np.random.rand(), s[gr][1] - am + 2 * am * np.random.rand()]

        # check if the randomly generated position is feasible

        # minimum distance between pedestrians
        d = []
        for l in range(i):      # go through all already generated pedestrians
            d.append(int(np.linalg.norm(pos - np.array(p[l][0:1])) <= r[i] + r[l]))

        # minimum distance from walls
        for l in range(num_walls):
            xp = pos[0]
            yp = pos[1]
            rp = np.array(pos)
            ra = obstacle_sides_arr[l, 0:2] 
            rb = obstacle_sides_arr[l, 2:4] 
            xa = ra[0]
            ya = ra[1]
            xb = rb[0]
            yb = rb[1]
            t = ((xp - xa) * (xb - xa) + (yp - ya) * (yb - ya)) / (((xb - xa) ** 2 + (yb - ya) ** 2))
            t_star = min(max(0, t), 1)
            rh = ra + t_star * (rb - ra)
            d.append(int(np.linalg.norm(rp - rh) <= r[i]))
        if sum(d) == 0:
            p[i] = [pos[0], pos[1], v[i, 0], v[i, 1], r[i], m[i]]
            X0 = np.append(X0, [pos[0], pos[1], th[i], np.linalg.norm(v[i, :]), 0, omg])
            i = i + 1

    return obstacle_sides_arr, num_walls, r, m , J, v0, v, th, omg, group_membership, X0, p

def agentsInitialization(n_groups, N, s, am, mapE, variablity, chaoticity):
    # return
    # r     Radius (N,1) np array
    # m     Mass (N,1) np array
    # J     Inertia (N,1) np array
    # v0    Desired speed (scalar)
    # v     Initial zeroed speeds (N,2) np array
    # th    Initial orientation (scalar)
    # omg   Initial angular velocity (scalar)
    # group_membership  Group_membership (N,1) np array
    # X0    Initial conditions for derivative variables (N, 6) np array
    #                                                   (N x (starting posX, starting posY, initial head orientation theta, size of the initial speed (0), 0, initial angular velocity))
    # p     position vectos (N, 6) np array
    #                       (N x (initial posX, initial posY, initial velocity x, initial velocity y, radius, mass))

    # Map loading
    num_walls, obstacle_sides_arr = mapE.obst_sides_arr.shape[0], mapE.obst_sides_arr

    group_membership = np.empty((N, 1), dtype=int)
    for i in range(len(n_groups)):  # random radii and masses
        # assign the i group to the relevant agents
        group_membership[sum(n_groups[0: i + 1]) - n_groups[i]: sum(n_groups[0: i + 1])] = int(i)

    agentsArr = np.zeros((N), dtype=object)
    groups = np.zeros(n_groups.size)

    for i in range(N):
        agentsArr[i] = Agent(i, int(group_membership[i]))
        agentsArr[i].v_desired += agentsArr[i].v_desired * variablity
        agentsArr[i].mass += agentsArr[i].mass * variablity
        agentsArr[i].radius += agentsArr[i].radius * variablity


    i = 0
    p = {}
    X0 = []

    # for each agent
    while i < N:
        gr = int(group_membership[i])
        if n_groups[agentsArr[i].group_id] > 1:
            agentsArr[i].isInGroup = True

        # generate a random starting point near the defined starting point
        pos = [s[gr][0] - am + 2 * am * np.random.rand(), s[gr][1] - am + 2 * am * np.random.rand()]

        # check if the randomly generated position is feasible

        # minimum distance between pedestrians
        d = []
        for l in range(i):      # go through all already generated pedestrians
            d.append(int(np.linalg.norm(pos - np.array(agentsArr[l].pos)) <= agentsArr[i].radius + agentsArr[l].radius))

        # minimum distance from walls
        for l in range(num_walls):
            nearestPoint = getNearestPointOnLine(obstacle_sides_arr[l, 0:2], obstacle_sides_arr[l, 2:4], pos)
            d.append(int(np.linalg.norm(np.array(pos) - nearestPoint) <= agentsArr[i].radius))
        if sum(d) == 0:
            agentsArr[i].setPos(pos)
            agentsArr[i].startingPos = pos
            X0 = np.append(X0, [pos[0], pos[1], agentsArr[i].theta, np.linalg.norm(agentsArr[i].init_speed), 0, agentsArr[i].angular_v])
            i = i + 1

    for i in range(len(n_groups)):
        id_s = sum(n_groups[0: i + 1]) - n_groups[i]
        id_end = sum(n_groups[0: i + 1])

        for agent in agentsArr[id_s : id_end]:
            agent.timeInStopped = abs(np.random.normal(2, 1))

    return obstacle_sides_arr, num_walls, group_membership, X0, agentsArr

class waypoints_updater():

    def __init__(self, e_seq, e_n, e_ind, e_act, N, n_groups, group_membership, isInterestDict, agentsLeft):
        self.e_seq = e_seq
        self.e_n = e_n          # num of waypoints for each group
        self.e_ind = e_ind      # current waypoint array
        self.e_act = e_act      # dictionary of np arrays (group_size, 2) with agents' current waypoint, one entry (/arr) for each group
        self.N = N
        self.group_membership = group_membership
        self.n_groups = n_groups
        self.isInterestDict = isInterestDict
        self.agentsLeft = agentsLeft

    def waypoint_update(self, position, coef, agents, t):
        e = np.zeros((self.N, 2))
        # Determination of the current waypoints
        for i in range(self.N):

            agents[i].checkInit(t)
            if agents[i].state == States.NOT_PRESENT:
               continue

            curr_wayp = self.e_act[agents[i].group_id] [i - sum(self.n_groups[0: agents[i].group_id ])]
            vect = curr_wayp - position[i]
            vect_norm = np.linalg.norm(curr_wayp - position[i])
            if vect_norm != 0:
                e[i, :] = vect / vect_norm
            else:
                e[i, :] = 0
            current_index = self.e_ind[ agents[i].group_id ][i - sum(self.n_groups[0: agents[i].group_id])]

            # agent has reached a waypoint && at least one more waypoint left to reach
            if (vect_norm <= coef and current_index < self.e_n[int(self.group_membership[i])]-1):
                current_index += 1
                curr_wayp = self.e_seq[agents[i].group_id][:, current_index].transpose()
                vect = curr_wayp - position[i]
                vect_norm = np.linalg.norm(curr_wayp - position[i])
                e[i, :] = vect / vect_norm

                self.e_ind[int(self.group_membership[i])][i - sum(self.n_groups[0:int(self.group_membership[i])])] = current_index
                self.e_act[int(self.group_membership[i])][i - sum(self.n_groups[0:int(self.group_membership[i])])] = curr_wayp

                if self.isInterestDict[agents[i].group_id][current_index-1]:
                    if agents[i].state == States.WALKING:
                        agents[i].stop()
                        agents[i].noteTime(t)
                        print("Agent ", i, "stops at time ", agents[i].timeStateEntered)
                    continue

            # agent has reached a waypoint && is at the end
            if vect_norm <= coef and current_index == self.e_n[int(self.group_membership[i])]-1:
                e[i, :] = ((1 - np.exp(-5 * vect_norm))/(1+np.exp(-5 * vect_norm))) * (vect / vect_norm)
                #print("at ", t, "loc: ", position[i], vect_norm, ", current waypoint at the end of agent ",  i, ": ", e[i, :])
                agents[i].disappear()
                self.agentsLeft -= 1;
                #continue

            # no waypoint has been reached
            if agents[i].state == States.STANDING:
                agents[i].checkStates(t)

            if agents[i].state == States.WALKING:
                agents[i].checkStates(t)

        return e        # returns a list of unit vectors pointing to the next waypoint


def get_group_center_of_mass(N, n_groups, group_membership, position):
    ci = {}
    for k in range(len(n_groups)):
        ci[k] = np.array([0, 0])

    for i in range(N):
        ci[int(group_membership[i])] = ci[int(group_membership[i])] + position[i]

    for k in range(len(n_groups)):
        ci[k] = ci[k] / n_groups[k]

    return ci

def write_to_file_sim_results(X, t, states):
    f = open("sim_results.txt", "w")
    f.close()

    t_entries = X.shape[0]
    f = open("sim_results.txt", "a")
    for lineIdx in range(t_entries):
        strTime = "{}".format(t[lineIdx][0])
        f.write(strTime)

        xPositions = X[lineIdx, :][0::6]
        yPositions = X[lineIdx, :][1::6]

        ang = X[lineIdx, :][2::6]
        velx = X[lineIdx, :][3::6]
        vely = X[lineIdx, :][4::6]


        agentStates = states[lineIdx, :]
        ids = np.arange(xPositions.size)

        line = ""
        for ag in range(xPositions.size):
            if agentStates[ag]:
                string = ";{},{},{}".format(ids[ag], xPositions[ag], yPositions[ag])
                line = line + string
        line = line + '\n'
        f.write(line)

    f.close()


def getStartEnd(filename):
    f = open(filename, "r")
    coords = np.zeros((2,2))

    coordsStart = f.readline()
    coords[0,:] = list(map(int, coordsStart.split(",")))
    coordsEnd = f.readline()
    coords[1, :] = list(map(int, coordsEnd.split(",")))
    return np.array(coords, dtype=int)
