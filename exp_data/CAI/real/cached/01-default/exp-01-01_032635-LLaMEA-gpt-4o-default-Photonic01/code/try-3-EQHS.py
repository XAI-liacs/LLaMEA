import numpy as np

class EQHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.3   # Pitch Adjusting Rate

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.budget, self.dim))
        self.harmony_values = np.full(self.budget, np.inf)

    def quantum_adjustment(self, candidate, best):
        return candidate + np.random.normal(0, 1, self.dim) * (best - candidate) / 2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmonies(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.budget):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < self.hmcr:
                    # Consider harmonies in memory
                    harmony = self.harmony_memory[np.random.randint(self.budget)]
                else:
                    # Randomly generate a new harmony
                    harmony = np.random.uniform(lb, ub, self.dim)

                # Pitch adjustment
                if np.random.rand() < self.par:
                    harmony = np.clip(self.quantum_adjustment(harmony, self.best_harmony if self.best_harmony is not None else harmony), lb, ub)

                current_value = func(harmony)
                evaluations += 1

                # Update harmony memory
                if current_value < self.harmony_values[i]:
                    self.harmony_values[i] = current_value
                    self.harmony_memory[i] = harmony.copy()

                # Update the best harmony
                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_harmony = harmony.copy()

        return self.best_harmony, self.best_value