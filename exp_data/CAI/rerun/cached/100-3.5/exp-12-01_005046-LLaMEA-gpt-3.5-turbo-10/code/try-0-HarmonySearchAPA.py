import numpy as np

class HarmonySearchAPA:
    def __init__(self, budget, dim, hmcr=0.7, par=0.4, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.dim,))

        def adjust_value(value, lower, upper):
            return np.clip(value, lower, upper)

        def explore_new_value(value, bw):
            return adjust_value(value + np.random.uniform(-bw, bw), -5.0, 5.0)

        def adjust_pitch_width(bw, iters):
            return bw * 0.99 ** iters

        harmony_memory = np.array([initialize_harmony_memory() for _ in range(10)])
        fitness_values = np.array([func(hm) for hm in harmony_memory])
        for _ in range(self.budget - 10):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.choice(len(harmony_memory))
                    new_solution[d] = harmony_memory[idx][d]
                    if np.random.rand() < self.par:
                        new_solution[d] = explore_new_value(new_solution[d], self.bw)
                else:
                    new_solution[d] = np.random.uniform(-5.0, 5.0)
            new_fitness = func(new_solution)
            if new_fitness < max(fitness_values):
                replace_idx = np.argmax(fitness_values)
                harmony_memory[replace_idx] = new_solution
                fitness_values[replace_idx] = new_fitness
            self.bw = adjust_pitch_width(self.bw, _)
        best_idx = np.argmin(fitness_values)
        return harmony_memory[best_idx]