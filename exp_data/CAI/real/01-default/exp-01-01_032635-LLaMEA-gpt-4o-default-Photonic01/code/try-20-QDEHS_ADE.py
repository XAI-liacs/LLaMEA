import numpy as np

class QDEHS_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.95
        self.par = 0.2
        self.local_search_rate = 0.25
        self.memory_size = min(50, budget)
        self.dynamic_par_step = 0.01
        self.F = 0.5
        self.CR = 0.9
        self.quantum_step_factor = 0.1  # Quantum diversity factor

    def initialize_harmonies(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        self.harmony_values = np.full(self.memory_size, np.inf)

    def quantum_adjustment(self, candidate, best, lb, ub):
        quantum_step = self.quantum_step_factor * np.random.uniform(-1, 1, self.dim)
        new_candidate = candidate + quantum_step * (best - candidate)
        return np.clip(new_candidate, lb, ub)

    def dynamic_pitch_adjustment(self, harmony, lb, ub):
        adjustment_strength = np.exp(-self.dim / self.memory_size)
        noise = np.random.uniform(-adjustment_strength, adjustment_strength, self.dim)
        new_harmony = harmony + noise
        self.par = min(0.5, self.par + self.dynamic_par_step * np.random.uniform(-1, 1))
        return np.clip(new_harmony, lb, ub)

    def differential_evolution(self, target, lb, ub):
        indices = np.random.choice(self.memory_size, 3, replace=False)
        x1, x2, x3 = self.harmony_memory[indices]
        mutant_vector = np.clip(x1 + self.F * (x2 - x3), lb, ub)
        crossover = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover, mutant_vector, target)
        return trial_vector

    def intensified_local_search(self, harmony, lb, ub):
        step_size = 0.05 * (ub - lb) * np.random.uniform(0.5, 1.5, self.dim)
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

                harmony = self.quantum_adjustment(harmony, self.best_harmony if self.best_harmony is not None else harmony, lb, ub)

                trial_vector = self.differential_evolution(harmony, lb, ub)
                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < self.harmony_values[i]:
                    self.harmony_values[i] = trial_value
                    self.harmony_memory[i] = trial_vector.copy()

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_harmony = trial_vector.copy()

        return self.best_harmony, self.best_value