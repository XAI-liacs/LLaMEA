import numpy as np

class QEDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_size = min(50, budget)
        self.positions = None
        self.energies = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.hmcr = 0.9  # Harmony memory consideration rate
        self.par = 0.3  # Pitch adjustment rate
        self.adapt_rate = 0.1
        self.bounds = None
        self.learning_rate = 0.1  # Learning rate for dynamic updates

    def initialize_harmony(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.harmony_size, self.dim))
        self.energies = np.full(self.harmony_size, np.inf)
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.harmony_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position, global_best):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim)
        new_position = position + beta * (global_best - position) + delta * 0.1
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def adaptive_harmony_search(self, position, lb, ub):
        perturbation = np.random.uniform(-self.par, self.par, self.dim)
        new_position = position + perturbation
        return np.clip(new_position, lb, ub)

    def adaptive_learning_strategy(self):
        for i in range(self.harmony_size):
            if np.random.rand() < self.learning_rate:
                self.positions[i] += np.random.uniform(-0.1, 0.1, self.dim)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmony(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.harmony_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.positions[i])
                evaluations += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.positions[i].copy()

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.positions[i].copy()

            self.adaptive_learning_strategy()

            for i in range(self.harmony_size):
                new_position = self.positions[i].copy()

                if np.random.rand() < self.hmcr:
                    random_index = np.random.randint(self.harmony_size)
                    new_position = self.positions[random_index].copy()
                
                if np.random.rand() < self.par:
                    new_position = self.adaptive_harmony_search(new_position, lb, ub)

                if np.random.rand() < self.adapt_rate:
                    new_position = self.quantum_position_update(new_position, self.global_best_position)

                new_value = func(new_position)
                if new_value < self.energies[i]:
                    self.positions[i] = new_position
                    self.energies[i] = new_value

        return self.global_best_position, self.global_best_value