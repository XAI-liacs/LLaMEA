import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.global_best_score = np.inf
        self.global_best_position = None
        self.particles = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.qbit_flip_prob = 0.5  # Probability of flipping a qubit
        self.expansion_coeff = 1.5  # Expansion coefficient for exploration

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _quantum_flip(self, position, lb, ub):
        new_position = np.copy(position)
        for j in range(self.dim):
            if np.random.rand() < self.qbit_flip_prob:
                new_position[j] = np.random.uniform(lb[j], ub[j])
        return new_position

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            # Quantum-inspired position update
            self.particles[i] = self._quantum_flip(self.particles[i], lb, ub)
            # Probabilistic expansion towards the global best
            if np.random.rand() < 0.5:
                self.particles[i] += self.expansion_coeff * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
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
                    self.personal_best_positions[i] = np.copy(self.particles[i])

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.particles[i])

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score