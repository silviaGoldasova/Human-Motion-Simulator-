import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import config
from aux_functions import parameters_load, initialization, waypoints_updater, agentsInitialization, write_to_file_sim_results, generateGroupDistribution, getStartEnd, readConfig
from HSFM_functions import HSFM_system
import agent
import plotting
from PRM_classes import PRMController, Obstacle, Map
from agent import Agent, AgentGroup
from PRM_classes import pathwayGeneration

numSamples = 700
numInterestPoints = 10
numGroupes = 12


def motion(TF, t_fine, uiInputs = None):

    tau, A, B, Aw, Bw, k1, k2, kd, ko, k1g, k2g, d_o, d_f, alpha, waypoint_reached_coef, am = parameters_load()
    generateNewData = False

    environmentFile, routesFile, startingPosFile, interestsFile = readConfig()

    if uiInputs != None:
        if uiInputs['mapFile'] != "":
            environmentFile = uiInputs['mapFile']
        variablity = uiInputs['variability']
        chaoticity = uiInputs['chaoticity']
        if uiInputs['mode1']:
            generateNewData = False
            if uiInputs['waypointsFile'] != "":
                routesFile = uiInputs['waypointsFile']
        else:
            generateNewData = True
            if uiInputs['startingPosFile'] != "":
                startingPosFile = uiInputs['startingPosFile']
            countWalkers = int(uiInputs['countWalkers'])
            groupFreq = int(uiInputs['groupFreq'])
    else:
        environmentFile = "Environment/environmentRealDataset.txt"
        routesFile = "Environment/pathsRealDataset.txt"
        interestsFile = "interests.txt"

    mapE = Map(environmentFile)

    if generateNewData:

        n_groups = generateGroupDistribution(countWalkers)
        N = sum(n_groups)  # Total number of individuals

        gateWayPoints = getStartEnd(startingPosFile)
        dict_waypoints, waypoint_counts, group_waypoint_idx, starting_points, isInterestDict = mapE.getRandomPaths(numInterestPoints, gateWayPoints, n_groups)
    else:
        # read data from a file
        routes, isInterestDict = pathwayGeneration.readPathFromFile(routesFile)
        numOfGroups = len(isInterestDict)

        # Number of individuals in each group => Define n_i the number of individuals in group i, then n_groups = [n_1, n_2, ..., n_N];
        n_groups = generateGroupDistribution(numOfGroups)
        N = sum(n_groups)  # Total number of individuals
        for i, key in enumerate(routes):
            mapE.plotPath(routes[key].T)
        dict_waypoints, waypoint_counts, group_waypoint_idx, starting_points = pathwayGeneration.get_paths_info(n_groups, routes)

    mapE.drawMap()

    # Initialization of agents
    map_walls, num_walls, group_membership, X0, agents = agentsInitialization(n_groups, N, starting_points, am, mapE, variablity, chaoticity)

    # Assign the actual position as the current waypoint
    e_act = {}
    for i in range(len(n_groups)):
        e_act[i] = np.zeros((n_groups[i], 2))   # current waypoint as point
    for i in range(N):
        e_act[int(group_membership[i])][i - sum(n_groups[0:int(group_membership[i])])] = agents[i].pos  # e_act[agent's_group_number][agent's_order_in_the_group] = starting pos of the agent

    agentsLeft = N
    config.waypoints = waypoints_updater(dict_waypoints, waypoint_counts, group_waypoint_idx, e_act, N, n_groups, group_membership, isInterestDict, agentsLeft)

    # System evolution
    sol = ode(HSFM_system).set_integrator('dopri5')
    t_start = 0.0
    t_final = TF
    delta_t = t_fine    # time accuracy
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((t_final - t_start)/delta_t) + 1)
    sol.set_initial_value(X0, t_start)
    states = np.zeros((num_steps, N), dtype=bool)
    statesTemp = np.zeros(N, dtype=bool) + True
    sol.set_f_params(N, n_groups, map_walls, num_walls, agents, group_membership, statesTemp)

    t = np.zeros((num_steps, 1))
    X = np.zeros((num_steps, N*6))
    t[0] = t_start
    X[0] = X0
    k = 1

    start_time = time.time()
    while sol.successful() and k < num_steps:
        sol.integrate(sol.t + delta_t)
        t[k] = sol.t
        X[k] = sol.y
        states[k, :] = statesTemp[:]
        statesTemp[:] = True
        k += 1
        if agentsLeft==0:
            break;

    elapsed_time = time.time() - start_time
    print("whole loop duration: ", elapsed_time, "frames: ", k)

    # Plotting
    plotting.plot_environment(X, N, n_groups.size, group_membership, mapE.allObs.shape[0], mapE.allObs, delta_t, states, 4)
    write_to_file_sim_results(X, t, states)

    return X, t, states, elapsed_time, k
