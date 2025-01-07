import numpy as np

class QuantumInspiredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30 + self.dim
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.2  # Quantum-inspired exploration parameter
        self.beta = 2.0  # Quantum-inspired exploitation parameter

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _quantum_update(self, lb, ub):
        for i in range(self.population_size):
            quantum_exploration = self.alpha * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            quantum_exploitation = self.beta * np.sin(self.velocities[i])
            self.velocities[i] *= np.cos(np.random.rand(self.dim) * np.pi)
            self.velocities[i] += quantum_exploration + quantum_exploitation
            self.particles[i] += self.velocities[i]
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