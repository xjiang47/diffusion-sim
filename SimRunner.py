import Initializer
import Diffuser
import Plotter
import numpy as np

class SimRunner:
    def __init__(self, initializer, diffuser, plotter):
        self.state, self.caps, self.tissue = initializer.initialize()
        self.live_tissue = self.tissue
        self.live_caps = self.caps
        self.diffuser = diffuser
        self.plotter = plotter
        self.curr_time = 0
        self.diffusion_percent = np.ones(self.state.shape, dtype=float)

    def run(self, timesteps = 100, interval = 1): # time step and interval defaults
        self.plotter.init()

        for i in range(timesteps):
            self.plotter.gather_data(self.state, self.live_caps, self.live_tissue, self.curr_time, self.diffusion_percent)
            self.state, self.live_tissue, self.live_caps, self.diffusion_percent = self.diffuser.diffuse(self.state, self.caps, self.tissue)
            if i % interval == 0:
                self.state, self.live_caps, self.live_tissue = self.plotter.interact(self.state, self.live_caps, self.live_tissue)
                diffuser.live_caps, diffuser.live_tissue = self.live_caps, self.live_tissue
            self.curr_time += diffuser.dt
        self.plotter.plot()

if __name__ == "__main__":
    cap = 0.0089285714
    tissue = 0.00892916955592076
    # the m values below are tuned for 40% saturation diffusing out of the capillaries and 60% remaining inside the capillaries
    # m values for 2x2, four way: m = .228
    # m values for 3x3, four way: m = .105
    # m values for 2x2, eight way: m = .2513
    # m values for 3x3, eight way: m = .098

    # the observed area, initial capillary value, initial tissue value, and fraction of capillaries "dead" (diffusing normally without replenishment) can be changed here.
    initializer = Initializer.Initializer3x3(rows=200, cols=200, cap=cap/4, tissue=cap/4, fraction_dead = .25) # to view the 2x2 layout, replace 3x3 with 2x2
    # the tissue death threshold, and live tissue consumption can be changed here.
    diffuser = Diffuser.DiffuserEightWay(initializer.tissue, initializer.caps,  initializer.cap_value, death_tissue=cap*.05, m=.098) # to view the four way diffusion, replace Eight with Four

    # Comment / uncomment out the plotters below to generate different plots
    plotter = Plotter.InteractivePlotter(cap=initializer.cap_value) # shows the visual evolution of the system over time,
    # capillaries can be turned off in real time, and time steps can be iterated through by pressing space

    # plotter = Plotter.PlotDeathsVTime()                             # shows both the capillary and tissue deaths over time
    # plotter = Plotter.ConservationPlotter()                         # shows the average value in all of the tissue cells
    # plotter = Plotter.CapConcPlotter(9,9)                           # shows the capillary concentration in any capillary, given its coordinates

    # note: to change the capillary death threshold, go into Diffuser.py, and change the percentage on the lines where there is the comment "change capillary threshold here"

    try:
        plotter.death_tissue = diffuser.death_tissue
    except Exception as e:
        pass
    sim = SimRunner(initializer, diffuser, plotter)
    sim.run(timesteps=1000, interval=10)            # make changes to the time step and timer interval here