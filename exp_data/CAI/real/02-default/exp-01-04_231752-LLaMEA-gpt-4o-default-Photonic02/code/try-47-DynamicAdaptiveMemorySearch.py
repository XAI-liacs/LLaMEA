import numpy as np

class DynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.memory_size = 5
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.omega = 0.5  # Initial inertia weight
        self.phi_p = 1.5  # Initial cognitive coefficient
        self.phi_g = 1.5  # Initial social coefficient
        self.memory = []
        self.performance_trend = []  # Track recent performance improvements

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            if self.memory and np.random.rand() < 0.5:
                strategy = self.memory[np.random.randint(len(self.memory))]
                cognitive_component = strategy['phi_p'] * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
                social_component = strategy['phi_g'] * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
                inertia = strategy['omega']
            else:
                cognitive_component = self.phi_p * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.phi_g * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
                inertia = self.omega

            self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
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
                    if len(self.memory) >= self.memory_size:
                        self.memory.pop(0)
                    self.memory.append({'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g})

                    self.performance_trend.append(score)
                    if len(self.performance_trend) > 10:
                        self.performance_trend.pop(0)

                    # Dynamic adaptation based on performance trend
                    if len(self.performance_trend) > 1:
                        trend = np.mean(np.diff(self.performance_trend))
                        if trend < 0:
                            self.omega = np.clip(self.omega + np.random.uniform(-0.05, 0.05), 0.3, 0.7)
                            self.phi_p = np.clip(self.phi_p + np.random.uniform(-0.05, 0.05), 1.2, 1.8)
                            self.phi_g = np.clip(self.phi_g + np.random.uniform(-0.05, 0.05), 1.2, 1.8)

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score