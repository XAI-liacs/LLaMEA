import numpy as np

class DynamicPitchHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.hmcr = 0.7
        self.par = 0.3
        self.bandwidth = 0.01

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def adjust_pitch(pitch, iteration):
            return pitch * np.exp(-iteration / self.budget)

        def explore_new_solution(pitch, harmony_memory, idx, iteration):
            new_solution = np.copy(harmony_memory[idx])
            for d in range(self.dim):
                if np.random.rand() < pitch:
                    new_solution[d] = np.random.uniform(self.lb, self.ub)
            return new_solution

        harmony_memory = initialize_harmony_memory()
        best_solution = harmony_memory[np.argmin([func(sol) for sol in harmony_memory])]
        iteration = 0
        while iteration < self.budget:
            new_harmony_memory = np.zeros((self.budget, self.dim))
            pitch = adjust_pitch(self.bandwidth, iteration)
            for i in range(self.budget):
                if np.random.rand() < self.hmcr:
                    idx = np.random.choice(self.budget)
                    new_harmony_memory[i] = explore_new_solution(pitch, harmony_memory, idx, iteration)
                else:
                    new_harmony_memory[i] = np.random.uniform(self.lb, self.ub, self.dim)
            harmony_memory = new_harmony_memory
            best_solution = harmony_memory[np.argmin([func(sol) for sol in harmony_memory])]
            iteration += 1
        return best_solution