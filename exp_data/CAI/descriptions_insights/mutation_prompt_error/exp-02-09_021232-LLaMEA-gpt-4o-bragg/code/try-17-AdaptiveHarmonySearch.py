import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3   # Pitch Adjustment Rate
        self.bandwidth = 0.05  # Initial Bandwidth
        self.num_harmonies = 10  # Number of harmonies in the memory
        self.harmony_memory = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_harmony_memory(self, bounds):
        self.harmony_memory = np.random.uniform(bounds.lb, bounds.ub, (self.num_harmonies, self.dim))
        self.harmony_memory_fitness = np.apply_along_axis(self.evaluate, 1, self.harmony_memory)

    def evaluate(self, func, harmony):
        return func(harmony)

    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_idx = np.argmax(self.harmony_memory_fitness)
        if new_fitness < self.harmony_memory_fitness[worst_idx]:
            self.harmony_memory[worst_idx] = new_harmony
            self.harmony_memory_fitness[worst_idx] = new_fitness

    def generate_new_harmony(self, bounds):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                new_harmony[i] = self.harmony_memory[np.random.randint(self.num_harmonies), i]
                if np.random.rand() < self.PAR:
                    new_harmony[i] += np.random.uniform(-1, 1) * self.bandwidth
            else:
                new_harmony[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def adjust_parameters(self):
        self.bandwidth *= np.random.uniform(0.9, 1.1)
        self.PAR = min(1.0, self.PAR + np.random.uniform(0.01, 0.03))
        # Dynamic adjustment for HMCR
        self.HMCR = max(0.7, self.HMCR - np.random.uniform(0, 0.01))

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmony_memory(bounds)

        evaluations = 0
        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(bounds)
            new_fitness = self.evaluate(func, new_harmony)
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_harmony.copy()
            self.update_harmony_memory(new_harmony, new_fitness)
            self.adjust_parameters()
            evaluations += 1

        return self.best_solution