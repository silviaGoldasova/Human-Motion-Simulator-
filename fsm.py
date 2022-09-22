from transitions import Machine
import enum

# The states
class States(enum.Enum):
    STANDING = 1
    FREEZED = 2
    ACCELERATING = 3
    NOT_PRESENT = 4
    DECELERATING = 5
    WALKING = 6

def createAgentFSM(agent):

    transitions = [
        ['stop', States.WALKING, States.STANDING],
        ['walk', States.STANDING, States.WALKING],
        ['freeze', States.WALKING, States.FREEZED],
        ['unfreeze', States.FREEZED, States.WALKING],
        ['speedUp', States.WALKING, States.ACCELERATING],
        ['speedDown', States.ACCELERATING, States.WALKING],
        ['disappear', States.WALKING, States.NOT_PRESENT],
        ['appear', States.NOT_PRESENT, States.WALKING],
    ]

    # Initialize
    machine = Machine(agent, states=States, transitions=transitions, initial=States.NOT_PRESENT)

    return machine