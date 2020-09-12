#!/usr/bin/python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backend_bases import MouseButton

# ctrl-. folds and unfolds code

class TissueSim:
    def __init__(self, rows=30, cols=30, d_k=9.5e2, m=0.0000580567, cap=0.0089285714, initial=0.00892916955592076, dt=1e-3, dx=1, ded=.6, interactive=True):
        # Set up constants
        self.D_K = d_k                      # Diffusion constant
        self.m = m                          # Consumption constant
        self.cap = cap                      # Capillary value
        self.initial = initial              # Initial tissue value
        self.dt = dt
        self.dx = dx
        self.K_t = d_k * (dt/(dx**2))
        self.K_b = 1+8 * d_k * (dt/(dx**2))
        self.ded = cap * ded
        self.deriv_filter = np.array([[1, 1., 1], [1, 0., 1], [1, 1., 1]])
        self.nbhr_avg_filter = np.array([[1, 1., 1], [1, 0., 1], [1, 1., 1]]) / 8
        self.curr_time = 0
        self.interactive = interactive

        # Set up graph
        self.fig, self.ax = plt.subplots()
        plt.ion()
        self.cid = self.fig.canvas.mpl_connect('button_release_event', self.kill_on_click)

        # Set up initial state
        cap_cell = np.array([[0, 0., 0], [0, 1., 0], [0, 0., 0]]) # defines 3x3 unit
        self.sim_state = np.tile(cap_cell, (rows, cols))
        self.cell_filt = self.sim_state == 0                      # tissue cells
        self.sim_state *= cap                                     # sets capillary concentrations
        self.live_tissue = self.cell_filt

        # show only
        self.ax.imshow(self.sim_state, norm=colors.Normalize(0, self.cap, clip=True), cmap="Reds")
        plt.draw()
        while not plt.waitforbuttonpress():
            pass
        if not interactive:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.cid = None

        tissue_conc = self.cell_filt * initial  # sets tissue concentrations
        self.sim_state += tissue_conc


    def kill_cap(self, x, y):
        self.sim_state[2*x+1, 2*y+1] = 0

    def kill(self, x, y):
        self.sim_state[x,y] = 0

    def calc_next_step(self):
        nbhr_avg = signal.convolve2d(self.sim_state, self.nbhr_avg_filter, mode="same", boundary="wrap")  # Finds average of neighboring units
        dead_caps = np.logical_and(nbhr_avg < self.ded, np.logical_not(self.cell_filt))  # Dead if neighboring average drops below threshold
        live_tissue = np.logical_and(self.sim_state > self.ded, self.cell_filt) # alive if value above threshold
        self.live_tissue = np.logical_and(self.live_tissue, live_tissue)
        self.sim_state[dead_caps] = 0

        deriv = signal.convolve2d(self.sim_state, self.deriv_filter, mode="same", boundary="wrap") * self.K_t  # Calculates discrete derivative
        self.sim_state[self.cell_filt] += deriv[self.cell_filt]
        self.sim_state[self.cell_filt] /= self.K_b
        self.sim_state[self.live_tissue] -= self.m * self.sim_state[self.live_tissue]
        self.curr_time += self.dt

    def plot(self):
        if not self.interactive:
            return
        self.ax.imshow(self.sim_state, norm=colors.Normalize(0, self.cap, clip=True), cmap="Reds")
        plt.draw()
        while not plt.waitforbuttonpress():
            pass

    def kill_on_click(self, event):
        if event.button == MouseButton.LEFT:
            if event.xdata is not None:
                self.kill(int(round(event.ydata)), int(round(event.xdata)))
                self.ax.imshow(self.sim_state, norm=colors.Normalize(0, self.cap, clip=True), cmap="Reds")
                plt.draw()

    def count_live_cells(self):
        return np.sum(self.live_tissue)

    def count_live_caps(self):
        return np.sum(np.logical_and(self.sim_state != 0, np.logical_not(self.cell_filt)))

    def get_current_time(self):
        return self.curr_time

if __name__ == "__main__":
    cap = 0.0089285714
    initial = 0.00892916955592076
    sim = TissueSim(cap = cap*.5, initial = initial * .5, ded = cap*.6) # , interactive=False
    live_cells = []
    live_caps = []
    time = []

    sim.plot() # Shows initial pre-diffusion state

    for i in range(100):
        for _ in range(10):
            sim.calc_next_step()
            live_cells.append(sim.count_live_cells())
            live_caps.append(sim.count_live_caps())
            time.append(sim.get_current_time())
        sim.plot()
    plt.figure()
    plt.plot(time, live_cells)
    plt.figure()
    plt.plot(time, live_caps)
    plt.show()
