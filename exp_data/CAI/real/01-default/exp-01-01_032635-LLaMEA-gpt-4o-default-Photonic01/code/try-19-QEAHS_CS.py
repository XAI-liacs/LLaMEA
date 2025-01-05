import numpy as np

class QEAHS_CS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(50, budget)
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.9
        self.par = 0.3
        self.F = 0.5
        self.CR = 0.7

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        self.harmony_values = np.array([np.inf] * self.memory_size)

    def quantum_perturbation(self, candidate):
        distance = np.random.normal(0, 1, self.dim)
        return candidate + distance * np.random.random(self.dim)

    def adaptive_adjustment(self, harmony, lb, ub):
        noise = np.random.uniform(-1, 1, self.dim) * (ub - lb) * 0.05
        return np.clip(harmony + noise, lb, ub)

    def coevolutionary_update(self, lb, ub):
        indices = np.random.choice(self.memory_size, 2, replace=False)
        x1, x2 = self.harmony_memory[indices]
        trial_vector = x1 + self.F * (x2 - x1)
        trial_vector = np.clip(trial_vector, lb, ub)
        return trial_vector

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmonies(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.memory_size):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < self.hmcr:
                    harmony = self.harmony_memory[np.random.randint(self.memory_size)]
                else:
                    harmony = np.random.uniform(lb, ub, self.dim)

                if np.random.rand() < self.par:
                    harmony = self.adaptive_adjustment(harmony, lb, ub)

                trial_vector = self.coevolutionary_update(lb, ub)
                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < self.harmony_values[i]:
                    self.harmony_values[i] = trial_value
                    self.harmony_memory[i] = trial_vector.copy()

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_harmony = trial_vector.copy()

        return self.best_harmony, self.best_value