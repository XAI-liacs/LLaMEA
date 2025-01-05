import numpy as np

class QuantumInspiredDynamicSelfAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_bits = np.random.rand(self.population_size, self.dim)  # Quantum bits for superposition
        self.collapsed_particles = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.zeros((self.population_size, self.dim))
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.collapsed_particles = (self.q_bits > 0.5) * (ub - lb) + lb
        self.personal_best_positions = np.copy(self.collapsed_particles)

    def _quantum_collapse(self, lb, ub):
        prob_threshold = np.random.rand(self.population_size, self.dim)
        self.collapsed_particles = (self.q_bits > prob_threshold) * (ub - lb) + lb
        self.collapsed_particles = np.clip(self.collapsed_particles, lb, ub)

    def _update_quantum_bits(self, lb, ub):
        for i in range(self.population_size):
            delta = (self.personal_best_positions[i] - self.collapsed_particles[i]) / (ub - lb)
            delta_global = (self.global_best_position - self.collapsed_particles[i]) / (ub - lb)
            self.q_bits[i] += 0.1 * delta + 0.1 * delta_global  # Dynamic update rule
            self.q_bits[i] = np.clip(self.q_bits[i], 0, 1)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            self._quantum_collapse(self.lb, self.ub)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.collapsed_particles[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.collapsed_particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.collapsed_particles[i]

            self._update_quantum_bits(self.lb, self.ub)

        return self.global_best_position, self.global_best_score