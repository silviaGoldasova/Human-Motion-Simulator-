import PySimpleGUI as sg
import launcher
import csv
import numpy as np
import matplotlib.pyplot as plt

def openWindow():
    sg.theme('DarkBlue')

    heading = [sg.Text('Enter Parameters for the Human Motion Simulation.')]
    selectModeText = [sg.Text('Select a Mode')]
    space = [sg.Text(' ' * 80)]
    spaceLonger = [sg.Text(' ' * 160)]
    buttons = [sg.Submit(), sg.Button("Exit")]

    layoutGeneralSection = [
        [sg.Text('Simulation time [s]', size=(15, 1)), sg.Slider(range=(5, 50), key='time', default_value=5, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text('Relative Chaoticity', size=(15, 1)), sg.Slider(range=(-5, 5), key='chaoticity', default_value=0, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text('Relative Variability', size=(15, 1)), sg.Slider(range=(-5, 5), key='variability', default_value=0, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text("Map Input File", size=(15, 1)), sg.Input(size=(20, 15), key="mapFile", pad=(2,15))],
    ]

    layoutMode1 = [
        [sg.Radio('Mode 1: Generate Simulation from Pregenerated Paths File', "Mode", key='mode1', default=True)],
        [sg.Text("Waypoints Input File", size=(15, 1)), sg.Input(size=(20, 1), key="waypointsFile")],
    ]

    layoutMode2 = [
        [sg.Radio('Mode 2: Generate New Paths for the Simulation', "Mode", key='mode2')],
        [sg.Text("Starting Positions Input File", size=(20, 1)), sg.Input(size=(30, 15), key="startingPosFile", pad=(2, 15))],
        [sg.Text('Number of Walkers', size=(20, 1)), sg.Slider(range=(1, 10), key='countWalkers', default_value=1, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text('Relative Group Frequency', size=(20, 1)), sg.Slider(range=(-5, 5), key='groupFreq', default_value=0, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
    ]

    layoutOptions = [
        [sg.Column(layoutMode1, vertical_alignment='top'), sg.HSeparator(pad=(30, 0)), sg.Column(layoutMode2, vertical_alignment='top')],
    ]

    layoutSliders = [
        [sg.Text('Desired speed [% change]', size=(15, 1)), sg.Slider(range=(-50, 50), key='desired_s', default_value=0, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text('Mass [% change]', size=(15, 1)), sg.Slider(range=(-50, 50), key='mass', default_value=0, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
        [sg.Text('Lateral movement [% change]', size=(15, 1)), sg.Slider(range=(-50, 50), key='lateral', default_value=0, size=(20, 15), orientation='horizontal', font=('Helvetica', 12))],
    ] 

    layout = [
        heading,
        layoutGeneralSection,
        spaceLonger,
        [sg.HSeparator(pad=(10, 30))],
        selectModeText,
        layoutOptions,
        space,
        [sg.HSeparator(pad=(10, 30))],
        buttons
    ]

    window = sg.Window('Human Motion Simulator', layout)

    while True:

        event, values = window.read()
        if event == "Submit":
            total_time = float(values['time'])
            t_fine = 1/10  # time accuracy
            X, t, states = launcher.motion(total_time, t_fine, values)
            print(values)

            continue
        if event == "Exit":
            break

if __name__ == '__main__':
    openWindow()
