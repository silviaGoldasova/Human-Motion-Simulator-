import time

import numpy as np
from aux_functions import parameters_load, getNearestPointOnLine, get_group_center_of_mass, getNearestPointOnLine
import config
from fsm import States

tmp = 0
min_person_interaction_distance = 2 # else the force is weak
min_obstacle_interaction_distance = 2.5

def HSFM_forces(X, e, N, map_walls, num_walls, agents):
    tau, A, B, Aw, Bw, k1, k2, kd, ko, k1g, k2g, d_o, d_f, alpha, waypoint_reached_coef, am = parameters_load()

    # Positions and velocities
    position = np.zeros((N, 2))
    vel = np.zeros((N, 2))
    for i in range(N):
        position[i, :] = [X[6 * i], X[6 * i+1]]
        vel[i, :] = [X[6 * i+3] * np.cos(X[6 * i+2]), X[6 * i +3] * np.sin(X[6 * i +2])]

    fi0 = np.zeros((N, 2))  # velocity force

    # Interindividual forces
    fij1 = np.zeros((N, 2))   # repulsive
    fij2 = np.zeros((N, 2))   # compression
    fij3 = np.zeros((N, 2))   # friction

    # Obstacles
    fiw1 = np.zeros((N, 2))   # repulsive
    fiw2 = np.zeros((N, 2))   # compression
    fiw3 = np.zeros((N, 2))   # friction
    ang = np.zeros((N,1))

    for i in range(N):

        fi0[i,:] = agents[i].mass * (agents[i].v_desired * e[i,:] - vel[i,:]) / tau
        vect = e[i,:]

        ang[i] = np.arctan2(vect[1], vect[0])

        for j in range(N):
            if i != j:
                dij = np.linalg.norm(position[i] - position[j])
                rij = agents[i].radius + agents[j].radius
                nij = (position[i] - position[j]) / dij
                fij1[i] = fij1[i] + A * np.exp((rij - dij) / B) * nij
                if dij < rij:
                    fij2[i] = fij2[i] + k1 * (rij - dij) * nij
                    tij = np.array([-nij[1], nij[0]])
                    dvij = np.dot((vel[j] - vel[i]), tij)
                    fij3[i] = fij3[i] + k2 * (rij - dij) * dvij * tij

        # Walls forces
        for w in range(num_walls):

            nearestPoint = getNearestPointOnLine(map_walls[w, 0:2], map_walls[w, 2:4], np.array(position[i,:]))

            diw = np.linalg.norm(position[i,:] - nearestPoint)
            niw = (position[i,:] - nearestPoint) / diw
            tiw = np.array([-niw[0], niw[1]])
            fiw1[i] = fiw1[i] + Aw * np.exp((agents[i].radius - diw) / Bw) * niw
            if diw < agents[i].radius:
                fiw2[i] = fiw2[i] + k1 * (agents[i].radius - diw) * niw
                fiw3[i] = fiw3[i] - k2 * (agents[i].radius - diw) * (vel[i] * tiw) * tiw

    # Force due to the desire to move as v0
    F1 = fi0

    # Other forces
    F2 = fij1 + fij2 + fij3 + fiw1 + fiw2 + fiw3

    return F1, F2, ang

def HSFM_system(t, X, N, n_groups, map_walls, num_walls, agents, group_membership, states):
    tau, A, B, Aw, Bw, k1, k2, kd, ko, k1g, k2g, d_o, d_f, alpha, waypoint_reached_coef, am = parameters_load()

    # Positions and velocities
    position = np.zeros((N, 2))
    vel = np.zeros((N, 2))

    for i in range(N):
        position[i, :] = [X[6 * i], X[6 * i + 1]]
        vel[i, :] = [X[6 * i + 3] * np.cos(X[6 * i + 2]), X[6 * i + 3] * np.sin(X[6 * i + 2])]

    e = config.waypoints.waypoint_update(position, waypoint_reached_coef, agents, t)

    dX = np.zeros((6*N,1)).flatten()

    # Acting forces
    F0, Fe, ang = HSFM_forces(X, e, N, map_walls, num_walls, agents)
    FT = F0 + Fe

    F_nV = (np.sqrt(np.sum(np.abs(F0) ** 2, 1)))

    #  desired theta
    thr = np.mod(ang, 2*np.pi).flatten()
    # actual theta
    th = np.mod(X.__getitem__(slice(2, None, 6)), 2*np.pi)

    # angle to rotate
    ang = np.unwrap(th - thr)

    # center of mass of each group: ci is a dictionary of np arrays (2,) holding positions of center of mass for each group
    ci = get_group_center_of_mass(N, n_groups, group_membership, position)

    for i in range(N):

        if agents[i].state == States.STANDING or agents[i].state == States.FREEZED or agents[i].state == States.NOT_PRESENT:
            if agents[i].state == States.NOT_PRESENT:
                states[i] = False
            continue

        dif = th[i] - thr[i]
        un = np.array([0, dif])
        angI = np.unwrap(un)[1]
        a = angI 

        kl = 0.3
        kth = agents[i].inertia * kl * F_nV[i]
        kom = agents[i].inertia * (1+alpha) * np.sqrt(kl * F_nV[i] / alpha)

        dX[6*i] = X[6*i+3] * np.cos(X[6*i+2]) - X[6*i+4] * np.sin(X[6*i+2])
        dX[6*i+1] = X[6*i+3] * np.sin(X[6*i+2]) + X[6*i+4] * np.cos(X[6*i+2])
        dX[6*i+2] = X[6*i+5]

        # Here we substitute the step function in the definition of the group cohesion forces with a sigmoid
        if n_groups[int(group_membership[i])] == 1: # is alone in the group => zero group cohesion force
            uf_group, uo_group = 0, 0
        else:
            p_i = ci[int(group_membership[i])] - position[i]
            uf_group = k1g * (1+np.tanh(5*(np.abs(np.dot(p_i, [np.cos(X[6*i+2]), np.sin(X[6*i+2])])-d_f)-3))) * np.dot(p_i / np.linalg.norm(p_i), [np.cos(X[6*i+2]), np.sin(X[6*i+2])])
            uo_group = k2g * (1+np.tanh(5*(np.abs(np.dot(p_i, [-np.sin(X[6*i+2]), np.cos(X[6*i+2])])-d_o)-3))) * np.dot(p_i / np.linalg.norm(p_i), [-np.sin(X[6*i+2]), np.cos(X[6*i+2])])

        dX[6*i+3] = 1 / agents[i].mass * (np.dot(FT[i], [np.cos(X[6*i+2]), np.sin(X[6*i+2])]) + uf_group)
        dX[6*i+4] = 1 / agents[i].mass * (ko*np.dot(Fe[i], [-np.sin(X[6*i+2]), np.cos(X[6*i+2])]) - kd * X[6*i+4] + uo_group)

        torque = -kth * a - kom * X[6*i+5]
        dX[6*i+5] = 1 / agents[i].inertia * torque  # d(angular velocity)/dt = angular acceleration/ = torque/inertia = torque/() )

    return dX
