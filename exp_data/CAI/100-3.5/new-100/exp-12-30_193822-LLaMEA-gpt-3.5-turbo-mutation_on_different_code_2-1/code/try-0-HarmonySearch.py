import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.2, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def generate_new_harmony(self, func, harmony_memory):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_harmony[i] = harmony_memory[np.random.choice(len(harmony_memory))][i]
            else:
                new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.uniform(-self.bw, self.bw)
            new_harmony[i] = np.clip(new_harmony[i], func.bounds.lb, func.bounds.ub)
        return new_harmony

    def __call__(self, func):
        harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.budget)]
        for i in range(self.budget):
            new_harmony = self.generate_new_harmony(func, harmony_memory)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = sorted(harmony_memory, key=lambda x: func(x))[:self.budget]
                harmony_memory[-1] = new_harmony
        return self.f_opt, self.x_opt