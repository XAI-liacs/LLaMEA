import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.alpha = 0.5  # Attraction parameter
        self.beta = 0.5   # Quantum parameter

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _quantum_update(self, position, lb, ub):
        # Quantum-inspired update using wave function model
        mbest = (self.alpha * self.global_best_position + (1 - self.alpha) * np.mean(self.personal_best_positions, axis=0))
        u = np.random.rand(self.dim)
        b = self.beta * np.random.uniform(lb, ub, self.dim)
        return mbest + b * np.sign(u - 0.5) * np.log(1.0 / (1.0 - u))

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.particles[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            for i in range(self.population_size):
                # Apply quantum-inspired update
                self.particles[i] = self._quantum_update(self.particles[i], self.lb, self.ub)
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

        return self.global_best_position, self.global_best_score