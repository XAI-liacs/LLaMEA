import numpy as np

class QuantumEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 1.49618  # cognitive parameter
        self.c2 = 1.49618  # social parameter
        self.w = 0.7298    # inertia weight

    def initialize_positions_and_velocities(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)

    def quantum_operator(self, particle_position, global_best):
        # Introduce quantum behavior by pulling particles closer to the global best
        quantum_step = np.random.uniform(-1, 1, self.dim) * np.log(1 + np.abs(global_best - particle_position))
        return particle_position + quantum_step

    def update_velocities_and_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)

    def __call__(self, func):
        self.initialize_positions_and_velocities(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                score = func(self.positions[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            self.update_velocities_and_positions(func.bounds)

            # Apply quantum operator occasionally to enhance exploration
            if evaluations % (self.population_size // 2) == 0:
                for i in range(self.population_size):
                    quantum_position = self.quantum_operator(self.positions[i], self.global_best_position)
                    quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
                    quantum_score = func(quantum_position)
                    evaluations += 1

                    if quantum_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = quantum_score
                        self.personal_best_positions[i] = quantum_position

                    if quantum_score < self.global_best_score:
                        self.global_best_score = quantum_score
                        self.global_best_position = quantum_position