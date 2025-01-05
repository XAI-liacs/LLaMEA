import numpy as np

class QuantumInspiredDynamicSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.particles = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.rotation_angle = np.pi / 4  # Quantum-inspired rotation angle
        self.dynamic_alpha = 0.1  # Parameter for dynamic adaptation

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _quantum_rotation_update(self, particle, personal_best, global_best):
        for d in range(self.dim):
            delta_theta = (np.random.rand() - 0.5) * 2 * self.rotation_angle  # Random delta angle
            rotation_matrix = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                        [np.sin(delta_theta), np.cos(delta_theta)]])
            position_vector = np.array([particle[d], global_best[d]])
            new_position = rotation_matrix @ position_vector
            particle[d] = (new_position[0] + personal_best[d]) / 2
        return particle

    def _dynamic_adjustment(self):
        diversity = np.std(self.particles, axis=0).mean()
        if diversity < 1e-5:  # Consider the population has converged
            self.rotation_angle = min(np.pi / 2, self.rotation_angle + self.dynamic_alpha)

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
                self.particles[i] = self._quantum_rotation_update(self.particles[i], 
                                                                  self.personal_best_positions[i], 
                                                                  self.global_best_position)
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

            self._dynamic_adjustment()

        return self.global_best_position, self.global_best_score