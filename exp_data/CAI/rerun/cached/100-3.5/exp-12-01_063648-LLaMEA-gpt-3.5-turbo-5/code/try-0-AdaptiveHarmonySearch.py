import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hmcr = 0.7
        self.par = 0.5
        self.bandwidth = 0.02
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = self.harmony_memory[np.random.randint(0, self.budget)]
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_solution[d] = new_solution[d] + np.random.uniform(-self.bandwidth, self.bandwidth)
                    else:
                        new_solution[d] = np.random.uniform(-5.0, 5.0)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[np.argsort([func(x) for x in self.harmony_memory])]
        return self.harmony_memory[0]