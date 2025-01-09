import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def generate_new_solution(self, func, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(len(new_harmony)):
            if np.random.rand() < self.hmcr:
                if np.random.rand() < self.par:
                    new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                else:
                    random_index = np.random.randint(0, len(harmony_memory))
                    new_harmony[i] = harmony_memory[random_index] + np.random.uniform(-self.bandwidth, self.bandwidth)
        return new_harmony

    def adapt_bandwidth(self, improvement):
        if improvement:
            self.bandwidth *= 2.0
        else:
            self.bandwidth *= 0.8

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        for _ in range(self.budget):
            new_harmony = self.generate_new_solution(func, harmony_memory)
            new_f = func(new_harmony)
            if new_f < self.f_opt:
                self.f_opt = new_f
                self.x_opt = new_harmony
                self.adapt_bandwidth(True)
            else:
                self.adapt_bandwidth(False)
            if new_f < func(harmony_memory):
                harmony_memory = new_harmony
        return self.f_opt, self.x_opt