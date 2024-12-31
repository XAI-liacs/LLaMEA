import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        bandwidth = self.bw

        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_harmony[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)

                if np.random.rand() < self.par:
                    new_harmony[d] += bandwidth * np.random.randn()

                new_harmony[d] = np.clip(new_harmony[d], func.bounds.lb, func.bounds.ub)

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony

            if i % 100 == 0:  # Adaptive bandwidth adjustment every 100 iterations
                bandwidth *= 0.9  # Reduce the bandwidth

            harmony_memory[i] = new_harmony

        return self.f_opt, self.x_opt