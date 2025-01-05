import numpy as np

class AQHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = min(50, budget)
        self.harmony_memory = None
        self.harmony_memory_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3  # Pitch Adjustment Rate
        self.fret_range = 0.05
        self.memory_decay = 0.98  # Reduce memory influence over time

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_harmony_sampling(self, lb, ub):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * self.fret_range
        new_harmony = self.best_harmony + beta * (np.mean(self.harmony_memory, axis=0) - self.best_harmony) + delta
        return np.clip(new_harmony, lb, ub)

    def adaptive_pitch_adjustment(self, harmony, lb, ub):
        if np.random.rand() < self.par:
            random_index = np.random.randint(0, self.dim)
            harmony[random_index] += np.random.uniform(-self.fret_range, self.fret_range)
        return np.clip(harmony, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmony_memory(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.harmony_memory_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.harmony_memory[i])
                evaluations += 1

                if current_value < self.harmony_memory_values[i]:
                    self.harmony_memory_values[i] = current_value
                    self.harmony_memory[i] = self.harmony_memory[i].copy()

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_harmony = self.harmony_memory[i].copy()

            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    index = np.random.randint(0, self.harmony_memory_size)
                    new_harmony[j] = self.harmony_memory[index][j]
                else:
                    new_harmony[j] = np.random.uniform(lb[j], ub[j])
            
            new_harmony = self.adaptive_pitch_adjustment(new_harmony, lb, ub)

            if np.random.rand() < (1 - self.memory_decay):
                new_harmony = self.quantum_harmony_sampling(lb, ub)

            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                max_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

                if new_value < self.best_value:
                    self.best_value = new_value
                    self.best_harmony = new_harmony

        return self.best_harmony, self.best_value