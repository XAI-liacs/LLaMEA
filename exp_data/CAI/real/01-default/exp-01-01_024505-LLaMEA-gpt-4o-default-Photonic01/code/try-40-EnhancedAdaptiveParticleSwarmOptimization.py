import numpy as np

class EnhancedAdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, min(50, budget // 10))
        self.particles = None
        self.velocities = None
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.best_personal_positions = None
        self.best_personal_fitness = np.full(self.swarm_size, float('inf'))
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2
        self.social_coeff = 2
        self.inertia_damping = 0.99

    def initialize_swarm(self, lb, ub):
        self.particles = lb + (ub - lb) * np.random.rand(self.swarm_size, self.dim)
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.best_personal_positions = self.particles.copy()

    def evaluate_particles(self, func):
        fitness = np.array([func(p) for p in self.particles])
        for i in range(self.swarm_size):
            if fitness[i] < self.best_personal_fitness[i]:
                self.best_personal_fitness[i] = fitness[i]
                self.best_personal_positions[i] = self.particles[i].copy()
        min_fitness_index = np.argmin(fitness)
        if fitness[min_fitness_index] < self.best_global_fitness:
            self.best_global_fitness = fitness[min_fitness_index]
            self.best_global_position = self.particles[min_fitness_index].copy()
        return fitness

    def update_velocities_and_positions(self, lb, ub):
        r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
        cognitive_component = self.cognitive_coeff * r1 * (self.best_personal_positions - self.particles)
        social_component = self.social_coeff * r2 * (self.best_global_position - self.particles)
        self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
        self.particles += self.velocities
        np.clip(self.particles, lb, ub, out=self.particles)
        self.inertia_weight *= self.inertia_damping

    def adapt_parameters(self, evaluations):
        self.cognitive_coeff = 2 - 1.5 * (evaluations / self.budget)
        self.social_coeff = 1.5 + 1.5 * (evaluations / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_particles(func)
            evaluations += len(fitness)

            if evaluations >= self.budget:
                break

            self.update_velocities_and_positions(lb, ub)
            self.adapt_parameters(evaluations)

        return self.best_global_position, self.best_global_fitness