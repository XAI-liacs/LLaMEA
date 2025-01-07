import numpy as np

class QuantumInspiredHarmonicSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hmsize = max(10, min(50, budget // 10))
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.harmony_memory = None
        self.best_harmony = None
        self.best_fitness = float('inf')

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = lb + (ub - lb) * np.random.rand(self.hmsize, self.dim)
        self.best_harmony = self.harmony_memory[0]

    def evaluate_harmony_memory(self, func):
        fitness = np.array([func(h) for h in self.harmony_memory])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_harmony = self.harmony_memory[i]
        return fitness

    def generate_new_harmony(self, lb, ub):
        new_harmony = np.copy(self.best_harmony)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_harmony[i] = self.harmony_memory[np.random.randint(self.hmsize)][i]
            if np.random.rand() < self.par:
                adjustment = (ub[i] - lb[i]) * (np.random.rand() - 0.5)
                new_harmony[i] += adjustment
                new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
        return new_harmony

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmony_memory(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_harmony_memory(func)
            evaluations += self.hmsize

            if evaluations >= self.budget:
                break

            new_harmony = self.generate_new_harmony(lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < max(fitness):
                worst_index = np.argmax(fitness)
                self.harmony_memory[worst_index] = new_harmony
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_harmony = new_harmony

        return self.best_harmony, self.best_fitness