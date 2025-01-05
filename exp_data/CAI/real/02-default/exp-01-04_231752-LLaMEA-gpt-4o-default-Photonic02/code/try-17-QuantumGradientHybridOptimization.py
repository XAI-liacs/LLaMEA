import numpy as np

class QuantumGradientHybridOptimization:
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
        self.omega = 0.5  # Inertia weight
        self.phi_p = 1.5  # Cognitive coefficient
        self.phi_g = 1.5  # Social coefficient
        self.tunneling_prob = 0.1  # Probability of quantum tunneling
        self.gradient_step = 0.05  # Step size for gradient adjustment

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _compute_gradient(self, particle, func, step_size):
        gradient = np.zeros(self.dim)
        for d in range(self.dim):
            step = np.zeros(self.dim)
            step[d] = step_size
            f_val_plus = func(np.clip(particle + step, self.lb, self.ub))
            f_val_minus = func(np.clip(particle - step, self.lb, self.ub))
            gradient[d] = (f_val_plus - f_val_minus) / (2 * step_size)
        return gradient

    def _update_particles(self, lb, ub, func):
        for i in range(self.population_size):
            cognitive_component = self.phi_p * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social_component = self.phi_g * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            self.velocities[i] = self.omega * self.velocities[i] + cognitive_component + social_component

            if np.random.rand() < self.tunneling_prob:
                self.particles[i] = self.global_best_position + np.random.normal(0, 0.1, self.dim) * (ub - lb)
            else:
                self.particles[i] += self.velocities[i] - self.gradient_step * self._compute_gradient(self.particles[i], func, self.gradient_step)

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
            
            self._update_particles(self.lb, self.ub, func)

        return self.global_best_position, self.global_best_score