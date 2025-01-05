import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hms=10, hmcr=0.7, par=0.4, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.hms, self.dim))

        def improvise_new_harmony(HM):
            new_harmony = np.zeros_like(HM[0])
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = HM[np.random.randint(self.hms)][i]
                    if np.random.rand() < self.par:
                        new_harmony[i] = new_harmony[i] + np.random.uniform(-self.bw, self.bw)
                        new_b = max(self.bw_range[0], min(self.bw_range[1], self.bw + np.random.normal(0, 0.1)))
                        self.bw = new_b
                else:
                    new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = improvise_new_harmony(harmony_memory)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            harmony_memory = np.vstack((harmony_memory, new_harmony))
            harmony_memory = harmony_memory[np.argsort([func(x) for x in harmony_memory])[:self.hms]]

        return self.f_opt, self.x_opt