import numpy as np

class EQHSPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr_initial = 0.9  # Initial Harmony Memory Considering Rate
        self.par_initial = 0.3   # Initial Pitch Adjusting Rate
        self.hmcr_final = 0.95   # Final Harmony Memory Considering Rate
        self.par_final = 0.1     # Final Pitch Adjusting Rate
        self.memory_size = min(100, budget)  # Initial size of the harmony memory

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        self.harmony_values = np.full(self.memory_size, np.inf)

    def quantum_adjustment(self, candidate, best):
        return candidate + np.random.normal(0, 1, self.dim) * (best - candidate) / 2

    def adapt_rates(self, evaluations):
        fraction = evaluations / self.budget
        self.hmcr = self.hmcr_initial + fraction * (self.hmcr_final - self.hmcr_initial)
        self.par = self.par_initial + fraction * (self.par_final - self.par_initial)

    def resize_population(self, evaluations):
        new_size = int(self.memory_size * (1 + evaluations / self.budget))
        if new_size > self.memory_size:
            self.harmony_memory = np.concatenate(
                [self.harmony_memory, np.random.uniform(lb, ub, (new_size - self.memory_size, self.dim))])
            self.harmony_values = np.concatenate(
                [self.harmony_values, np.full(new_size - self.memory_size, np.inf)])
            self.memory_size = new_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmonies(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.adapt_rates(evaluations)
            self.resize_population(evaluations)

            for i in range(self.memory_size):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < self.hmcr:
                    # Consider harmonies in memory
                    harmony = self.harmony_memory[np.random.randint(self.memory_size)]
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