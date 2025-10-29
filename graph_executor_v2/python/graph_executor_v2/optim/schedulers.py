# graph_executor_v2/optim/schedulers.py
import math

class StepLR:
    def __init__(self, optimizer, step_size: int, gamma: float=0.1):
        self.opt = optimizer; self.step_size = step_size; self.gamma = gamma; self.t = 0
    def step(self):
        self.t += 1
        if self.t % self.step_size == 0:
            for g in self.opt.param_groups:
                g["lr"] *= self.gamma

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max: int, eta_min: float=0.0):
        self.opt = optimizer; self.T_max=T_max; self.eta_min=eta_min; self.t=0
        self.base = [g["lr"] for g in optimizer.param_groups]
    def step(self):
        self.t += 1
        for base_lr, g in zip(self.base, self.opt.param_groups):
            g["lr"] = self.eta_min + 0.5*(base_lr - self.eta_min)*(1+math.cos(math.pi*self.t/self.T_max))
