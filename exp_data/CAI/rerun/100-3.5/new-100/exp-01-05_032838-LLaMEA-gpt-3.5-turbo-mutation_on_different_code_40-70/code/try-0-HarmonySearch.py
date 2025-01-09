import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hms=10, hmcr=0.9, par=0.4, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.hms, self.dim))

        for _ in range(self.budget):
            new_harmony = np.array([self._generate_new_harmony(harmony_memory, func) for _ in range(self.hms)])
            harmony_memory = np.vstack((harmony_memory, new_harmony))

            harmony_memory = harmony_memory[np.argsort([func(x) for x in harmony_memory])[:self.hms]]

        self.x_opt = harmony_memory[0]
        self.f_opt = func(self.x_opt)

        return self.f_opt, self.x_opt

    def _generate_new_harmony(self, harmony_memory, func):
        new_harmony = np.array([self._select_value(harmony_memory, i) for i in range(self.dim)])
        for i in range(self.dim):
            if np.random.uniform() < self.par:
                new_harmony[i] = harmony_memory[np.random.randint(0, self.hms)][i]
            if np.random.uniform() < self.bw:
                new_harmony[i] += np.random.uniform(-self.bw, self.bw)

        return np.clip(new_harmony, func.bounds.lb, func.bounds.ub)

    def _select_value(self, harmony_memory, idx):
        if np.random.uniform() < self.hmcr:
            return harmony_memory[np.random.randint(0, self.hms)][idx]
        else:
            return np.random.uniform(func.bounds.lb, func.bounds.ub)