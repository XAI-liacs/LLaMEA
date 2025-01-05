import numpy as np

class QHS_AMR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.9  # Harmonized memory consideration rate
        self.par = 0.2   # Pitch adjusting rate
        self.local_search_rate = 0.3  # Probability of local search
        self.memory_size = min(50, budget)
        self.dynamic_par_step = 0.02

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        self.harmony_values = np.full(self.memory_size, np.inf)

    def quantum_adjustment(self, candidate, best):
        return candidate + np.random.normal(0, 1, self.dim) * (best - candidate) / 1.5

    def dynamic_pitch_adjustment(self, harmony, lb, ub):
        adjustment_strength = np.exp(-self.dim / self.memory_size)
        noise = np.random.uniform(-adjustment_strength, adjustment_strength, self.dim)
        new_harmony = harmony + noise
        self.par = min(0.5, self.par + self.dynamic_par_step * np.random.uniform(-1, 1))
        return np.clip(new_harmony, lb, ub)

    def intensified_local_search(self, harmony, lb, ub):
        step_size = 0.03 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        new_harmony = harmony + perturbation
        return np.clip(new_harmony, lb, ub)

    def adaptive_local_search(self, harmony, lb, ub, iteration, max_iterations):
        step_size = (0.02 + 0.03 * (1 - iteration / max_iterations)) * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        new_harmony = harmony + perturbation
        return np.clip(new_harmony, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmonies(lb, ub)
        evaluations = 0
        max_iterations = self.budget // self.memory_size

        for iteration in range(max_iterations):
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
                    if np.random.rand() < 0.5:
                        harmony = self.intensified_local_search(harmony, lb, ub)
                    else:
                        harmony = self.adaptive_local_search(harmony, lb, ub, iteration, max_iterations)

                current_value = func(harmony)
                evaluations += 1

                if current_value < self.harmony_values[i]:
                    self.harmony_values[i] = current_value
                    self.harmony_memory[i] = harmony.copy()

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_harmony = harmony.copy()

        return self.best_harmony, self.best_value