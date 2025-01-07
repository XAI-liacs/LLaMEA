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
        # Initialize particles' positions using quantum superposition principle
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

                # Quantum-inspired velocity update
                quantum_cognition = (personal_best_positions[i] - positions[i]) * np.random.rand(self.dim)
                quantum_social = (global_best_position - positions[i]) * np.random.rand(self.dim)
                velocities[i] = self.w * velocities[i] + self.c1 * quantum_cognition + self.c2 * quantum_social

            # Update positions and handle boundaries
            positions += velocities
            positions = np.clip(positions, lb, ub)
            # Adaptive inertia weight update
            self.w = 0.4 + 0.3 * (self.budget - evals) / self.budget
            self.c2 = 1.5 + 0.5 * (self.budget - evals) / self.budget  # Dynamic social coefficient

        return global_best_position, global_best_score