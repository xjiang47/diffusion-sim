import numpy as np
from scipy import signal

class DiffuserFourWay:
    def __init__(self, tissue, caps, cap_initial, d_k=950, m=0.0000580567, dt=1e-3, dx=1, death_tissue = 0.0004465):
        # facilitates four way diffusion
        self.d_k = d_k                            # diffusion constant, micrometers squared per second
        self.m = m                                # consumption, concentration per time step
        self.dt = dt                              # time step, seconds
        self.dx = dx                              # distance step, micrometers squared
        self.live_caps = caps                     # live caps
        self.live_tissue = tissue                 # live tissue
        self.cap_initial = cap_initial            # defines initial capillary value, millimoles per milliliter
        self.death_tissue = death_tissue          # tissue death threshold
        self.K_t = d_k * (dt / (dx ** 2))
        self.K_b = 1 + 4 * d_k * (dt / (dx ** 2))

        self.deriv_filter = np.array([[0, 1., 0], [1, 0., 1], [0, 1., 0]]) # defines cross filter

    def diffuse(self, state, caps, tissue):
        live_tissue = np.logical_and(state > self.death_tissue, tissue)  # tissue is alive if its value is above a threshold
        self.live_tissue = np.logical_and(self.live_tissue, live_tissue) # dead tissue stays dead

        # Calculates discrete derivative
        old_state = state                                                                            # saves the previous state
        deriv = signal.convolve2d(state, self.deriv_filter, mode="same", boundary="wrap") * self.K_t # applies filter and creates periodic boundary conditions
        state += deriv                                                                               # delta is added to the previous state
        state /= self.K_b
        diffusion_percent = state/old_state                                                          # calculates percent change in state due to diffusion
        state[self.live_tissue] -= self.m*old_state[self.live_tissue]                                # consumption is applied to live tissue

        live_caps = state > .5 * self.cap_initial                  # capillaries are alive above 50% saturation @todo: note this line
        # print(np.mean((state/self.cap_initial)[self.live_caps]))   # prints percent saturation
        self.live_caps = np.logical_and(self.live_caps, live_caps) # dead capillaries stay dead
        state[self.live_caps] = self.cap_initial                   # live capillaries are refilled

        return state, self.live_tissue, self.live_caps, diffusion_percent

class DiffuserEightWay:
    def __init__(self, tissue, caps, cap_initial, d_k=950, m=0.0000580567, dt=1e-3, dx=1, death_tissue = 0.0004465):
        # facilitates eight way diffusion
        self.d_k = d_k                            # diffusion constant, micrometers squared per second
        self.m = m                                # consumption, concentration per time step
        self.dt = dt                              # time step, seconds
        self.dx = dx                              # distance step, micrometers squared
        self.live_caps = caps                     # live caps
        self.live_tissue = tissue                 # live tissue
        self.cap_initial = cap_initial            # defines initial capillary value, millimoles per milliliter
        self.death_tissue = death_tissue          # tissue death threshold
        self.K_t = d_k * (dt / (dx ** 2))
        self.K_b = 1 + 6 * d_k * (dt / (dx ** 2))

        self.deriv_filter = np.array([[.5, 1., .5], [1, 0., 1], [.5, 1., .5]]) # defines filter for surrounding cells

    def diffuse(self, state, caps, tissue):
        live_tissue = np.logical_and(state > self.death_tissue, tissue)  # tissue is alive if its value is above a threshold
        self.live_tissue = np.logical_and(self.live_tissue, live_tissue) # dead tissue stays dead

        # Calculates discrete derivative
        old_state = state.copy()                                                                     # saves the previous state
        deriv = signal.convolve2d(state, self.deriv_filter, mode="same", boundary="wrap") * self.K_t # applies filter and creates periodic boundary conditions
        state += deriv                                                                               # delta is added to the previous state
        state /= self.K_b
        diffusion_percent = state / old_state  # calculates percent change in state due to diffusion
        state[self.live_tissue] -= self.m * old_state[self.live_tissue]                              # consumption is applied to live tissue


        live_caps = state > .5 * self.cap_initial                  # capillaries are alive above 50% saturation @todo: note this line
        # print(np.mean((state / self.cap_initial)[self.live_caps])) # prints percent saturation
        self.live_caps = np.logical_and(self.live_caps, live_caps) # dead capillaries stay dead
        state[self.live_caps] = self.cap_initial                   # live capillaries are refilled

        return state, self.live_tissue, self.live_caps, diffusion_percent

