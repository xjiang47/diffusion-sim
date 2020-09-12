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

    def run(self, timesteps = 100, interval = 1):
        self.plotter.init()

        for i in range(timesteps):
            self.plotter.gather_data(self.state, self.live_caps, self.live_tissue, self.curr_time, self.diffusion_percent)
            self.state, self.live_tissue, self.live_caps, self.diffusion_percent = self.diffuser.diffuse(self.state, self.caps, self.tissue)
            if i % interval == 0:
                self.state, self.live_caps, self.live_tissue = self.plotter.interact(self.state, self.live_caps, self.live_tissue)
                diffuser.live_caps, diffuser.live_tissue = self.live_caps, self.live_tissue
                print(self.state[5, 5])
                print(0.00893*.05)
            self.curr_time += diffuser.dt
        self.plotter.plot()

if __name__ == "__main__":
    cap = 0.0089285714
    tissue = 0.00892916955592076
    # m values for 2x2, four way: m = .228
    # m values for 3x3, four way: m = .105
    # m values for 2x2, eight way: m = .2513
    # m values for 3x3, eight way: m = .098
    initializer = Initializer.Initializer3x3(rows=200, cols=200, cap=cap/4, tissue=cap/4, fraction_dead = .25)
    diffuser = Diffuser.DiffuserEightWay(initializer.tissue, initializer.caps,  initializer.cap_value, death_tissue=cap*.05, m=.098)
    plotter = Plotter.InteractivePlotter(cap=initializer.cap_value)
    # plotter = Plotter.PlotDeathsVTime()
    # plotter = Plotter.ConservationPlotter()
    # plotter = Plotter.CapConcPlotter(9,9)
    # plotter = Plotter.CapConcPlotter(10, 10)

    try:
        plotter.death_tissue = diffuser.death_tissue
    except Exception as e:
        pass
    sim = SimRunner(initializer, diffuser, plotter)
    sim.run(timesteps=1000, interval=10)