import numpy as np

class QEAHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.25  # Pitch Adjusting Rate
        self.local_search_rate = 0.3  # Probability of applying local search
        self.memory_size = min(50, budget)
        self.beta = 0.5  # Probability of quantum adjustment

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        self.harmony_values = np.full(self.memory_size, np.inf)

    def quantum_superposition(self, candidate, best):
        direction = best - candidate
        step = np.random.uniform(-self.beta, self.beta, self.dim) * direction
        return candidate + step

    def dynamic_pitch_adjustment(self, harmony, lb, ub):
        noise = np.random.uniform(-1, 1, self.dim)
        new_harmony = harmony + self.par * noise * (ub - lb)
        return np.clip(new_harmony, lb, ub)

    def intensified_local_search(self, harmony, lb, ub):
        step_size = 0.1 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        new_harmony = harmony + perturbation
        return np.clip(new_harmony, lb, ub)

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

                if np.random.rand() < self.beta:
                    harmony = self.quantum_superposition(harmony, self.best_harmony if self.best_harmony is not None else harmony)

                if np.random.rand() < self.par:
                    harmony = self.dynamic_pitch_adjustment(harmony, lb, ub)

                if np.random.rand() < self.local_search_rate:
                    harmony = self.intensified_local_search(harmony, lb, ub)

                current_value = func(harmony)
                evaluations += 1

                if current_value < self.harmony_values[i]:
                    self.harmony_values[i] = current_value
                    self.harmony_memory[i] = harmony.copy()

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_harmony = harmony.copy()

        return self.best_harmony, self.best_value