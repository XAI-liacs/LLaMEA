import numpy as np

class QMHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_size = min(50, budget)
        self.harmonies = None
        self.harmony_memory = None
        self.harmony_values = None
        self.best_harmony = None
        self.best_value = np.inf
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.fw = 0.02   # Fret Width for pitch adjustment
        self.adapt_rate = 0.1
        self.bounds = None

    def initialize_harmonies(self, lb, ub):
        self.harmonies = np.random.uniform(lb, ub, (self.harmony_size, self.dim))
        self.harmony_memory = self.harmonies.copy()
        self.harmony_values = np.full(self.harmony_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, harmony, best_harmony):
        beta = np.random.normal(0, 1, self.dim)
        new_harmony = harmony + beta * (best_harmony - harmony)
        lb, ub = self.bounds
        return np.clip(new_harmony, lb, ub)

    def pitch_adjustment(self, harmony, lb, ub):
        adjustment = np.random.uniform(-self.fw, self.fw, self.dim)
        return np.clip(harmony + adjustment, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmonies(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.harmony_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.harmonies[i])
                evaluations += 1

                if current_value < self.harmony_values[i]:
                    self.harmony_values[i] = current_value
                    self.harmony_memory[i] = self.harmonies[i].copy()

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_harmony = self.harmonies[i].copy()

            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = self.harmony_memory[np.random.randint(self.harmony_size)][d]
                    if np.random.rand() < self.par:
                        new_harmony[d] = self.pitch_adjustment(new_harmony, lb, ub)[d]
                else:
                    new_harmony[d] = np.random.uniform(lb[d], ub[d])

            if np.random.rand() < self.adapt_rate:
                new_harmony = self.quantum_position_update(new_harmony, self.best_harmony)

            new_value = func(new_harmony)
            evaluations += 1

            if new_value < self.best_value:
                self.best_value = new_value
                self.best_harmony = new_harmony.copy()

        return self.best_harmony, self.best_value