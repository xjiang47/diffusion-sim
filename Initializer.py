import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# sets up 2x2 unit to be replicated to generate our observational area
class Initializer2x2:
    def __init__(self, rows=30, cols=30, cap=0.00893, tissue=0.00892857, fraction_dead = 0):
        # Set up initial state
        cap_cell = np.array([[1., 0], [0., 0]])              # defines 2x2 unit
        self.initial_state = np.tile(cap_cell, (rows, cols)) # tiles an area with the rows by cols number of units
        self.tissue = self.initial_state == 0                # tissue cells are where the array equals 0

        # Set up graph
        self.fig, self.ax = plt.subplots()
        plt.ion()
        cid = self.fig.canvas.mpl_connect('button_release_event', self.kill_on_click)
        self.ax.imshow(self.initial_state, cmap="Reds")
        plt.draw()
        while not plt.waitforbuttonpress():
            pass
        plt.close(self.fig)
        plt.ioff()

        self.caps = self.initial_state == 1     # cap cells are where the cells are not tissue
        cap_indicies = np.argwhere(self.caps == 1)
        killed_indicies = cap_indicies[np.random.choice(len(cap_indicies), int(fraction_dead*len(cap_indicies)), replace = False)]
        self.caps[killed_indicies[:,0], killed_indicies[:,1]] = 0
        self.cap_value = cap                    # stores capillary concentration
        self.tissue_value = tissue              # stores tissue concentration
        self.initial_state *= cap               # sets capillary concentrations
        tissue_conc = self.tissue * tissue      # sets tissue concentrations
        self.initial_state += tissue_conc       # combines capillaries and tissue into a single array

    def initialize(self):
        # initializes with the initial state, capillaries, and tissue
        return self.initial_state, np.logical_not(self.tissue), self.tissue

    def kill(self, x, y):
        # sets killed coordinate to 0
        self.initial_state[x,y] = 0

    def kill_on_click(self, event):
        if event.button == MouseButton.LEFT:
            if event.xdata is not None:
                self.kill(int(round(event.ydata)), int(round(event.xdata)))
                self.ax.imshow(self.initial_state, cmap="Reds")
                plt.draw()

# sets up 3x3 unit to be replicated to generate our observational area
class Initializer3x3:
    def __init__(self, rows=30, cols=30, cap=0.00893, tissue=0.00892857, fraction_dead = 0):
        # Set up initial state
        cap_cell = np.array([[0, 0., 0], [0, 1., 0], [0, 0., 0]])    # defines 3x3 unit
        self.initial_state = np.tile(cap_cell, (rows, cols))         # tiles an area with the rows by cols number of units, cuts top row and left column
        self.tissue = self.initial_state == 0                        # tissue cells are where the array equals 0

        # Set up graph
        self.fig, self.ax = plt.subplots()
        plt.ion()
        cid = self.fig.canvas.mpl_connect('button_release_event', self.kill_on_click)
        self.ax.imshow(self.initial_state, cmap="Reds")
        plt.draw()
        while not plt.waitforbuttonpress():
            pass
        plt.close(self.fig)
        plt.ioff()

        self.caps = self.initial_state == 1  # cap cells are where the cells are not tissue
        cap_indicies = np.argwhere(self.caps == 1)
        # print(cap_indicies)
        killed_indicies = cap_indicies[np.random.choice(len(cap_indicies), int(fraction_dead * len(cap_indicies)), replace=False)]
        # print(killed_indicies)
        self.caps[killed_indicies[:,0], killed_indicies[:,1]] = 0
        self.cap_value = cap  # stores capillary concentration
        self.tissue_value = tissue  # stores tissue concentration
        self.initial_state *= cap  # sets capillary concentrations
        tissue_conc = self.tissue * tissue  # sets tissue concentrations
        self.initial_state += tissue_conc  # combines capillaries and tissue into a single array

    def initialize(self):
        # initializes with the initial state, capillaries, and tissue
        return self.initial_state, np.logical_not(self.tissue), self.tissue

    def kill(self, x, y):
        # sets killed coordinate to 0
        self.initial_state[x,y] = 0

    def kill_on_click(self, event):
        if event.button == MouseButton.LEFT:
            if event.xdata is not None:
                self.kill(int(round(event.ydata)), int(round(event.xdata)))
                self.ax.imshow(self.initial_state, cmap="Reds")
                plt.draw()