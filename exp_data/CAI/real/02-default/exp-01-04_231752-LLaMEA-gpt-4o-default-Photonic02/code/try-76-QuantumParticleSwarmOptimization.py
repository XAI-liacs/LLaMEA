import numpy as np

class QuantumParticleSwarmOptimization:
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
        self.alpha = 0.5  # Contraction-expansion coefficient
        self.beta = 1.5  # Attraction strength

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.random.rand(self.dim) * (ub - lb) + lb

    def _quantum_update(self, lb, ub):
        for i in range(self.population_size):
            phi = np.random.rand(self.dim)
            p = phi * self.personal_best_positions[i] + (1 - phi) * self.global_best_position
            u = np.random.rand(self.dim) - 0.5
            self.particles[i] = p + self.alpha * np.abs(self.global_best_position - self.particles[i]) * np.log(1/u)
            self.particles[i] = np.clip(self.particles[i], lb, ub)

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

            self._quantum_update(self.lb, self.ub)

        return self.global_best_position, self.global_best_score