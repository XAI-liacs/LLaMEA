import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(30, budget // 10)
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight

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
                fitness_value = func(positions[i])
                evals += 1
                if fitness_value < personal_best_scores[i]:
                    personal_best_scores[i] = fitness_value
                    personal_best_positions[i] = positions[i].copy()
                if fitness_value < global_best_score:
                    global_best_score = fitness_value
                    global_best_position = positions[i].copy()

                quantum_cognition = (personal_best_positions[i] - positions[i]) * np.random.rand(self.dim) * np.sin((evals / self.budget * np.pi) + np.pi/6) * (1 + 0.1 * np.sin(evals * np.pi / self.budget))  # Modified line
                quantum_social = (global_best_position - positions[i]) * np.random.rand(self.dim) * np.sin(evals / self.budget * np.pi)
                velocities[i] = self.w * velocities[i] + self.c1 * quantum_cognition + self.c2 * quantum_social + np.random.normal(0, 0.1, self.dim) + np.power(evals / self.budget, 2)

            positions += velocities * (0.8 + 0.2 * np.sin((evals / self.budget * np.pi) + np.pi/4))

            positions = np.clip(positions, lb, ub)
            self.w = 0.5 + 0.2 * (self.budget - evals) / self.budget
            self.c1 = 1.6 + 0.4 * (self.budget - evals) / self.budget
            self.c2 = 1.5 + 0.6 * (self.budget - evals) / self.budget

        return global_best_position, global_best_score