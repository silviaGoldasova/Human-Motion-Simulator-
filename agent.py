import numpy as np
from fsm import States, createAgentFSM
import time

radius_u = 0.3
radius_sd = 0.05

prob_female = 0.5
prob_male = 0.5

mass_men_u = 90.0
mass_female_u = 74.1
mass_men_sd = 15.8
mass_female_sd = 16.2

v0_male_u=1.27
v0_male_sd=0.21

v0_female_u=1.19
v0_female_sd=0.19

v0 = 1.19

eventsArr = np.array([States.WALKING, States.ACCELERATING, States.FREEZED])
probsArr = np.array([0.98, 0.01, 0.01])

timeInAccel = 2

def getEventGivenProbs(eventsArr, probsArr):
    if np.sum(probsArr) != 1:
        probsArr /= np.sum(probsArr)
    cumProbsArr = np.cumsum(probsArr)

    randomNum = np.random.rand()
    for i, prob in enumerate(cumProbsArr):
        if randomNum < prob:
            return eventsArr[i]

# Individual characteristics
def setAgentsCharacteristics():
    # Radius
    rm = 0.25  # minimum radius
    rM = 0.35  # maximum radius

    # Mass
    mm = 60  # minimum mass
    mM = 90  # maximum mass

    # Desired speed 1-1.2 => male: u=1.27, sd=0.21; female: u=1.19, sd=0.19
    v0m = 1  # minimum speed
    v0M = 1.2  # maximum speed

    return rm, rM, mm, mM, v0m, v0M


class Agent:
    def __init__(self, i, group_id=-1):
        self.i = i
        self.machine = createAgentFSM(self)
        self.radius = np.random.normal(radius_u, radius_sd)
        self.sex = getEventGivenProbs(['f', 'm'], [prob_female, prob_male])
        self.timeInStopped = 0

        if self.sex == 'f':
            self.mass = np.random.normal(mass_female_u, mass_female_sd)
            self.v_desired = np.random.normal(v0_female_u, v0_female_sd)
        else:
            self.mass = np.random.normal(mass_men_u, mass_men_sd)
            self.v_desired = np.random.normal(v0_male_u, v0_male_sd)

        self.startingPos = None
        self.init_speed = 0        # v
        self.theta = np.pi  # or pi
        self.angular_v = 0      # omv
        self.inertia = 0.5 * self.radius ** 2  # Inertia #* self.mass
        self.group_id = group_id
        self.isInGroup = False
        self.timeLeftInTheState = 2
        self.initAppearanceTime = np.random.randint(low=0, high=4)

    def setPos(self, pos):
        self.pos = np.array(pos)

    def on_enter_STANDING(self):
        print(self.i, ": entered state standing, for time t=", self.timeInStopped)
        self.timeStateEntered = time.time()

    def on_enter_WALKING(self):
        print(self.i, ":entered state walking!",)
        self.timeStateEntered = time.time()

    def on_enter_FREEZED(self):
        self.timeStateEntered = time.time()
        print(self.i, ": entered state freezed!")

    def on_enter_ACCELERATING(self):
        self.timeStateEntered = time.time()
        self.v_desired = 2 * self.v_desired
        print(self.i, ": entered state accelerating!")

    def on_exit_ACCELERATING(self):
        self.v_desired = self.v_desired / 2
        print(self.i, " the end of state accelerating")

    def on_enter_NOT_PRESENT(self):
        print(self.i, " entered state not present")

    def checkInit(self, t):
        if t > self.initAppearanceTime and self.state == States.NOT_PRESENT and t < self.initAppearanceTime+1:
            print(self.i, ": agent has appeared", t)
            self.appear()

    def checkStates(self, t):
        if (t - self.timeStateEntered) > self.timeInStopped:
            if (self.state == States.STANDING or self.state == States.FREEZED):
                self.walk()
                print(self.i, ": walking again at time ", t)

        if (t - self.timeStateEntered) > timeInAccel:
            if self.state == States.ACCELERATING:
                self.speedDown()

    def randomChangeState(self):
        if self.state == States.WALKING:
            newEvent = getEventGivenProbs(eventsArr, probsArr)
            if newEvent == States.ACCELERATING:
                self.speedUp()
            elif newEvent == States.FREEZED:
                self.freeze()

    def noteTime(self, t):
        self.timeStateEntered = t

    def print(self):
        attrs = vars(self)  # returns a dictionary
        print(', '.join("%s: %s" % item for item in attrs.items()))

    #def getRandTimeStopped(self):


class AgentGroup:

    def __init__(self, i):
        self.id = i

    def initialize_group_characteristics(self):
        return


# Individual characteristics
def setAgentsCharacteristicsSimple():
    # Radius
    rm = 0.25  # minimum radius
    rM = 0.35  # maximum radius

    # Mass
    mm = 60  # minimum mass
    mM = 90  # maximum mass

    # Desired speed 1-1.2 => male: u=1.27, sd=0.21; female: u=1.19, sd=0.19
    v0m = 1  # minimum speed
    v0M = 1.2  # maximum speed

    return rm, rM, mm, mM, v0m, v0M

