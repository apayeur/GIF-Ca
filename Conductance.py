import numpy as np

class Conductance :
    #Attributes:
    #   tau_rise = rise time (ms)
    #   tau_decay = decay time (ms)
    #   g_max = maximal conductance (nS)
    def __init__(self, tau_rise=1., tau_decay=5., g_max=0.1):
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.g_max = g_max

    def set_g_max(self, val):
        self.g_max = val

    def get_g_max(self):
        return self.g_max

    def get(self, t):
        t_max = self.tau_decay * self.tau_rise / (self.tau_rise - self.tau_decay) * np.log(self.tau_rise / self.tau_decay)
        A = self.g_max / (np.exp(-t_max / self.tau_decay) - np.exp(-t_max / self.tau_rise))
        return A * (np.exp(-t / self.tau_decay) - np.exp(-t / self.tau_rise)) * np.heaviside(t, 0.)

    @staticmethod
    def magnesium_block(V):
        return 1./(1. + np.exp(-V/16.13)/3.57)