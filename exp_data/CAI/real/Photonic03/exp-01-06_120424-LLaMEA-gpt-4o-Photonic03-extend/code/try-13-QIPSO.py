import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(30, budget // 10)
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight
        self.initial_population_size = self.population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([float('inf')] * self.population_size)
        global_best_position = None
        global_best_score = float('inf')

        evals = 0
        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                fitness_value = func(positions[i])
                evals += 1
                if fitness_value < personal_best_scores[i]:
                    personal_best_scores[i] = fitness_value
                    personal_best_positions[i] = positions[i].copy()
                if fitness_value < global_best_score:
                    global_best_score = fitness_value
                    global_best_position = positions[i].copy()

                quantum_cognition = (personal_best_positions[i] - positions[i]) * np.random.rand(self.dim)
                quantum_social = (global_best_position - positions[i]) * np.random.rand(self.dim)
                velocities[i] = self.w * velocities[i] + self.c1 * quantum_cognition + self.c2 * quantum_social

            positions += velocities * (0.9 + 0.1 * np.random.rand())
            positions = np.clip(positions, lb, ub)
            self.w = 0.4 + 0.3 * (self.budget - evals) / self.budget
            self.c1 = 1.5 + 0.5 * (self.budget - evals) / self.budget
            self.c2 = 1.5 + 0.5 * (self.budget - evals) / self.budget

            # Dynamic adaptation of the population size
            self.population_size = max(10, int(self.initial_population_size * (1 - evals / self.budget)))

        return global_best_position, global_best_score