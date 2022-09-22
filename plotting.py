import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter
import random

def getRandomColor():
    color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color

def get_arr_random_colors(number_of_colors):
    color_arr = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    return color_arr

def plot_png(X, N, num_walls, map_walls, group_membership, arr_colors):
    plt.figure()

    # Plot of the walls
    for i in range(num_walls):
        if map_walls[i].isWall:
            x = [item[0] for item in map_walls[i].allCords]
            y = [item[1] for item in map_walls[i].allCords]
            plt.scatter(x, y, c="k")
            plt.plot(x, y)
        else:
            plt.gca().add_patch(matplotlib.patches.Rectangle((map_walls[i].bottomLeft[0], map_walls[i].bottomLeft[1]), map_walls[i].width, map_walls[i].height, alpha=0.4))

    # Starting points
    plt.plot(X[0].__getitem__(slice(0, None, 6)), X[0].__getitem__(slice(1, None, 6)), 'ro')

    # Trajectories
    for i in range(N):
        plt.plot(X[:, 6 * i], X[:, 6 * i + 1], color=arr_colors[int(group_membership[i])])

    plt.axis('equal')
    plt.savefig('trajectories.png')


def plot_environment(X, N, num_of_groups, group_membership, num_walls, map_walls, interval_param, agents, desired_total_time=0):
    arr_colors = get_arr_random_colors(num_of_groups)

    # plot and save image
    plot_png(X, N, num_walls, map_walls, group_membership, arr_colors)

    x = np.array(X[:, 0::6])
    y = np.array(X[:, 1::6])

    # plot animation
    plt = plotter(x, y, interval_param, desired_total_time, num_walls, map_walls, arr_colors, group_membership, agents)

    plt.generate_anim(agents)

class plotter():

    def __init__(self, xdataP, ydataP, interval_param, desired_total_time, num_walls, map_walls, color_arr, group_membership, agents):
        self.xdata = xdataP
        self.ydata = ydataP
        self.data_len, self.N = self.xdata.shape     # num of frames x num of agents
        self.anim = None
        self.interval_t = 0
        self.color_arr = color_arr
        self.group_membership = group_membership

        if desired_total_time == 0: # keep the real speed
            self.interval_t = interval_param * 1000
        else:                       # speed up the anim
            self.interval_t = ( float(desired_total_time) / self.data_len) * 1000

        if self.interval_t < 0:
            self.interval_t = 1
        self.interval_t = int(self.interval_t)

        self.fig = plt.figure()
        self.axis = plt.axes()
        self.lines = [self.axis.plot([], [], color=self.color_arr[int(self.group_membership[i])], marker='o')[0] for i in range(self.N)]     # for each agent

        # Plot of the walls
        for i in range(num_walls):
            if map_walls[i].isWall:
                x = [item[0] for item in map_walls[i].allCords]
                y = [item[1] for item in map_walls[i].allCords]
                plt.scatter(x, y, c="k")
                plt.plot(x, y)
            else:
                plt.gca().add_patch(matplotlib.patches.Rectangle((map_walls[i].bottomLeft[0], map_walls[i].bottomLeft[1]), map_walls[i].width, map_walls[i].height, alpha=0.4))

        # End points
        plt.plot(self.xdata[0,:], self.ydata[0,:], 'o')

        plt.gca().set_aspect('equal', adjustable='box')


    def generate_anim(self, agents, anim_name=None):

        def init_plot():
            for line in self.lines:
                line.set_data([], [])
            return self.lines

        # animation function
        def animate(i):
            for j,line in enumerate(self.lines):
                if agents[i,j]:
                    line.set_data(self.xdata[i,j], self.ydata[i,j]) #  facecolors='none', edgecolors=self.color_arr[int(self.group_membership[j])]
            return self.lines

        # calling the animation function
        self.anim = FuncAnimation(self.fig, animate, init_func=init_plot, frames=self.data_len, interval=self.interval_t, blit=True)
        self.anim.save("scenario_.gif", dpi=300, writer=PillowWriter(fps=35))

        plt.show()

    @staticmethod
    def getRandomColor():
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        return color