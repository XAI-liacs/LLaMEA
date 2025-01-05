import numpy as np

class AdvancedQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, min(50, budget // 10))
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = np.full(self.num_particles, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.cognitive_const = 2.0
        self.social_const = 2.0
        self.inertia_weight = 0.7
        self.constriction_factor = 0.5
        self.dynamic_neighborhood_size = 3
        self.interference_prob = 0.1
        self.levy_prob = 0.2
        self.memory_factor = 0.1
        self.reinit_threshold = 5
        self.learning_rate = 0.1  # Initial learning rate for adaptive control

    def initialize_particles(self, lb, ub):
        self.particles = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.randn(self.num_particles, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.last_improvement = 0

    def evaluate_particles(self, func):
        fitness = np.array([func(p) for p in self.particles])
        for i, f in enumerate(fitness):
            if f < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = f
                self.personal_best_positions[i] = self.particles[i]
            if f < self.global_best_fitness:
                self.global_best_fitness = f
                self.global_best_position = self.particles[i]
                self.last_improvement = 0
        return fitness

    def update_velocities_and_positions(self, lb, ub):
        for i in range(self.num_particles):
            local_best = self.get_local_best(i)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_term = self.learning_rate * self.cognitive_const * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_term = self.learning_rate * self.social_const * r2 * (local_best - self.particles[i])
            adaptive_memory = self.memory_factor * (self.global_best_position - self.particles[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_term + social_term + adaptive_memory)
            self.particles[i] += self.constriction_factor * self.velocities[i]
            if np.random.rand() < self.levy_prob:
                self.levy_flight(i, lb, ub)
        self.particles = np.clip(self.particles, lb, ub)

    def get_local_best(self, index):
        neighborhood_indices = np.random.choice(self.num_particles, self.dynamic_neighborhood_size, replace=False)
        local_best_index = min(neighborhood_indices, key=lambda i: self.personal_best_fitness[i])
        return self.personal_best_positions[local_best_index]

    def levy_flight(self, i, lb, ub):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / np.abs(v) ** (1 / beta)
        self.particles[i] += 0.01 * step * (self.particles[i] - self.global_best_position)
        self.particles[i] = np.clip(self.particles[i], lb, ub)

    def apply_quantum_interference(self, lb, ub):
        for i in range(self.num_particles):
            if np.random.rand() < self.interference_prob:
                interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.particles[i] = np.mean([self.particles[i], interference_vector], axis=0)
                self.particles[i] = np.clip(self.particles[i], lb, ub)

    def stochastic_velocity_reinit(self):
        if self.last_improvement >= self.reinit_threshold:
            indices = np.random.choice(self.num_particles, self.num_particles // 3, replace=False)
            for i in indices:
                self.velocities[i] = np.random.randn(self.dim) * 0.1
            self.last_improvement = 0

    def adapt_learning_rate(self, evaluations):
        max_eval = self.budget
        self.learning_rate = 0.1 + (0.9 * (1 - evaluations / max_eval))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_particles(func)
            evaluations += self.num_particles

            if evaluations >= self.budget:
                break

            self.adapt_learning_rate(evaluations)
            self.update_velocities_and_positions(lb, ub)
            self.apply_quantum_interference(lb, ub)
            self.stochastic_velocity_reinit()
            self.last_improvement += 1

        return self.global_best_position, self.global_best_fitness