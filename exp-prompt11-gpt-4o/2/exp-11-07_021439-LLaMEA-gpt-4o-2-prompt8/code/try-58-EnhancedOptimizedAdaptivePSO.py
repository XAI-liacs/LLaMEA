import numpy as np

class EnhancedOptimizedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.n_particles = 30
        self.c1 = 2.05
        self.c2 = 2.05
        self.inertia_weight = 0.9
        self.inertia_damping = 0.99
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.2
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, dim))
        self.velocities = np.zeros((self.n_particles, dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.n_particles, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.rng = np.random.default_rng()

    def __call__(self, func):
        func_calls = 0
        inertia_weight = self.inertia_weight
        random_cache = self.rng.random((2, self.n_particles, self.dim, int(self.budget / self.n_particles)))
        iteration = 0
        while func_calls < self.budget:
            fitness_values = np.array([func(p) for p in self.positions])
            func_calls += self.n_particles
            
            improved = fitness_values < self.pbest_scores
            self.pbest_scores[improved] = fitness_values[improved]
            self.pbest_positions[improved] = self.positions[improved]
            
            min_index = np.argmin(self.pbest_scores)
            if self.pbest_scores[min_index] < self.gbest_score:
                self.gbest_score = self.pbest_scores[min_index]
                self.gbest_position = self.pbest_positions[min_index]
            
            r1, r2 = random_cache[0, :, :, iteration % random_cache.shape[3]], random_cache[1, :, :, iteration % random_cache.shape[3]]
            cognitive_term = self.c1 * r1 * (self.pbest_positions - self.positions)
            social_term = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = inertia_weight * self.velocities + cognitive_term + social_term
            np.clip(self.velocities, -self.max_velocity, self.max_velocity, out=self.velocities)
            self.positions += self.velocities
            np.clip(self.positions, self.lower_bound, self.upper_bound, out=self.positions)
            
            inertia_weight *= self.inertia_damping
            iteration += 1