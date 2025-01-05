import numpy as np

class EAMQHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.95  # Increased Harmony Memory Considering Rate
        self.par = 0.15   # Reduced Pitch Adjusting Rate for finer adjustments
        self.local_search_rate = 0.2  # Increased probability of applying local search
        self.memory_size = min(50, budget)  # Fixed memory size for better management
        self.dynamic_par_step = 0.01  # Dynamic adjustment for pitch rate

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        self.harmony_values = np.full(self.memory_size, np.inf)

    def quantum_adjustment(self, candidate, best):
        return candidate + np.random.normal(0, 1, self.dim) * (best - candidate) / 2

    def dynamic_pitch_adjustment(self, harmony, lb, ub):
        adjustment_strength = np.exp(-self.dim / self.memory_size)
        noise = np.random.uniform(-adjustment_strength, adjustment_strength, self.dim)
        new_harmony = harmony + noise
        self.par = min(0.5, self.par + self.dynamic_par_step * np.random.uniform(-1, 1))  # Dynamic adjustment
        return np.clip(new_harmony, lb, ub)

    def intensified_local_search(self, harmony, lb, ub):
        step_size = 0.05 * (ub - lb)  # Smaller step size for intensified search
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