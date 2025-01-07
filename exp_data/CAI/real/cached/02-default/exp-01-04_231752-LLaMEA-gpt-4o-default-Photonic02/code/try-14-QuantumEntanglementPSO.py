import numpy as np

class QuantumEntanglementPSO:
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
        self.tunneling_prob = 0.1  # Initial probability of quantum tunneling
        self.adaptive_factor = 0.1  # Adaptive factor for tunneling probability

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            cognitive_component = self.phi_p * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social_component = self.phi_g * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            self.velocities[i] = self.omega * self.velocities[i] + cognitive_component + social_component

            if np.random.rand() < self.tunneling_prob:
                # Quantum tunneling: randomize particle position with a bias towards the global best
                self.particles[i] = self.global_best_position + np.random.normal(0, 0.1, self.dim) * (ub - lb)
            else:
                self.particles[i] += self.velocities[i]

            # Entangle particles to share global best information
            if np.random.rand() < 0.05:  # Small probability to entangle
                entangled_particle = np.random.choice(self.population_size)
                self.particles[i] = (self.particles[i] + self.particles[entangled_particle]) / 2

            self.particles[i] = np.clip(self.particles[i], lb, ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        prev_global_best_score = np.inf
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

            # Adaptive tunneling adjustment
            if self.global_best_score < prev_global_best_score:
                self.tunneling_prob = max(0.01, self.tunneling_prob - self.adaptive_factor)
            else:
                self.tunneling_prob = min(0.3, self.tunneling_prob + self.adaptive_factor)
            prev_global_best_score = self.global_best_score

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score