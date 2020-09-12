import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.backend_bases import MouseButton

class PlotDeathsVTime:
    def __init__(self):
        self.time = []              # defines list that tracks time
        self.live_cap_count = []    # defines list that tracks live capillaries
        self.live_tissue_count = [] # defines list that tracks live tissue

    def init(self):
        pass

    def gather_data(self, state, live_cap, live_tissue, time):
        self.time.append(time)                             # adds current time to time list
        self.live_cap_count.append(np.sum(live_cap))       # adds current live capillaries to live capillaries list
        self.live_tissue_count.append(np.sum(live_tissue)) # adds current live tissue to live tissue list

    def plot(self):
        # plots live tissue vs time
        plt.figure()
        plt.plot(self.time, self.live_tissue_count)
        plt.title("Live tissue count vs Time")
        # plots live capillary vs time
        plt.figure()
        plt.plot(self.time, self.live_cap_count)
        plt.title("Live capillary count vs Time")
        plt.show()

    def interact(self, state, live_cap, live_tissue):
        return state, live_cap, live_tissue


class InteractivePlotter:
    def __init__(self, cap=0.00893):
        self.fig, self.ax = plt.subplots()
        self.cap = cap
        self.state = None
        self.live_cap = None
        self.live_tissue = None

    def init(self):
        plt.ion()

    def gather_data(self, state, live_cap, live_tissue, time, diffusion_percent):
        pass

    def interact(self, state, live_cap, live_tissue):
        self.state = state
        self.live_cap = live_cap
        self.live_tissue = live_tissue
        # interactive plotting loop here
        self.ax.imshow(self.state, norm=colors.Normalize(0, self.cap, clip=True), cmap="Reds")
        cid = self.fig.canvas.mpl_connect('button_release_event', self.kill_on_click)
        plt.draw()
        while not plt.waitforbuttonpress():
            pass
        self.fig.canvas.mpl_disconnect(cid)

        return self.state, self.live_cap, self.live_tissue

    def plot(self):
        plt.ioff()

    def kill_on_click(self, event):
        if event.button == MouseButton.LEFT:
            if event.xdata is not None:
                self.kill_cap(int(round(event.ydata)), int(round(event.xdata)))
                self.ax.imshow(self.state, norm=colors.Normalize(0, self.cap, clip=True), cmap="Reds")
                plt.draw()

    def kill_cap(self, x, y):
        self.live_cap[x,y] = False

class ConservationPlotter:
    def __init__(self):
        self.average = [] # defines list that tracks average tissue concentration
        self.time = []    # defines list that tracks time

    def init(self):
        pass

    def gather_data(self, state, live_cap, live_tissue, time, diffusion_percent):
        self.average.append(np.sum(state[live_tissue])/np.sum(live_tissue)) # adds current tissue average to average tissue concentration list
        self.time.append(time)                                              # adds current time to time list

    def plot(self):
        # plots average tissue concentration vs time
        plt.figure()
        plt.plot(self.time, self.average)
        plt.title("Average Tissue Oxygen Concentration vs Time")
        plt.show()

    def interact(self, state, live_cap, live_tissue):
        return state, live_cap, live_tissue

class CapConcPlotter:
    def __init__(self, r, c):
        self.cap_conc = [] # defines list that tracks average tissue concentration
        self.time = []     # defines list that tracks time
        self.r = r
        self.c = c

    def init(self):
        pass

    def gather_data(self, state, live_cap, live_tissue, time, diffusion_percent):
        self.cap_conc.append(diffusion_percent[self.r, self.c]) # adds current tissue average to average tissue concentration list
        self.time.append(time)                                  # adds current time to time list

    def plot(self):
        # plots average tissue concentration vs time
        plt.figure()
        plt.plot(self.time, self.cap_conc)
        plt.title("Average Tissue Oxygen Concentration vs Time")
        plt.show()

    def interact(self, state, live_cap, live_tissue):
        return state, live_cap, live_tissue