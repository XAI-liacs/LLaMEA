import numpy as np

class AdaptiveMemeticQPSO:
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
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.inertia_weight = 0.6
        self.levy_prob = 0.3
        self.mutation_prob = 0.1
        self.diversification_factor = 0.05

    def initialize_particles(self, lb, ub):
        self.particles = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.randn(self.num_particles, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)

    def evaluate_particles(self, func):
        fitness = np.array([func(p) for p in self.particles])
        for i, f in enumerate(fitness):
            if f < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = f
                self.personal_best_positions[i] = self.particles[i]
            if f < self.global_best_fitness:
                self.global_best_fitness = f
                self.global_best_position = self.particles[i]
        return fitness

    def update_velocities_and_positions(self, lb, ub):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_term = self.cognitive_const * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_term = self.social_const * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] + cognitive_term + social_term)
            self.particles[i] += self.velocities[i]
            if np.random.rand() < self.levy_prob:
                self.levy_flight(i)
            if np.random.rand() < self.mutation_prob:
                self.mutate_particle(i, lb, ub)
        self.particles = np.clip(self.particles, lb, ub)

    def levy_flight(self, i):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / np.abs(v) ** (1 / beta)
        self.particles[i] += 0.01 * step

    def mutate_particle(self, i, lb, ub):
        perturbation = self.diversification_factor * (ub - lb) * (np.random.rand(self.dim) - 0.5)
        self.particles[i] += perturbation
        self.particles[i] = np.clip(self.particles[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_particles(func)
            evaluations += self.num_particles

            if evaluations >= self.budget:
                break

            self.update_velocities_and_positions(lb, ub)

        return self.global_best_position, self.global_best_fitness