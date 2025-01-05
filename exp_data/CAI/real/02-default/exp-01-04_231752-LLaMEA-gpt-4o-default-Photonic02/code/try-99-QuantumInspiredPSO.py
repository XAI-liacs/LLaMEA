import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + dim
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.75  # Alpha coefficient for quantum behavior
        self.beta = 0.25   # Beta coefficient for quantum behavior

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _quantum_update(self, lb, ub):
        # Quantum-inspired update using probabilistic superposition
        mean_position = np.mean(self.particles, axis=0)
        for i in range(self.population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            quantum_move = np.exp(-self.beta * r1) * (mean_position - self.particles[i])
            random_move = self.alpha * (r2 - 0.5)
            self.particles[i] += self.velocities[i] + quantum_move + random_move
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