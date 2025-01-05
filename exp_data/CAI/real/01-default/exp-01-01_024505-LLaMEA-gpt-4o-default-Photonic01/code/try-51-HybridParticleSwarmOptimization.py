import numpy as np

class HybridParticleSwarmOptimization:
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
        self.inertia_weight = 0.7
        self.crossover_prob = 0.8
        self.mutation_prob = 0.1

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
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_term + social_term)
            self.particles[i] += self.velocities[i]
        self.particles = np.clip(self.particles, lb, ub)

    def crossover(self, lb, ub):
        for i in range(0, self.num_particles, 2):
            if np.random.rand() < self.crossover_prob:
                crossover_point = np.random.randint(0, self.dim)
                parent1, parent2 = self.particles[i], self.particles[(i+1) % self.num_particles]
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                self.particles[i], self.particles[(i+1) % self.num_particles] = child1, child2
        self.particles = np.clip(self.particles, lb, ub)

    def mutate(self, lb, ub):
        for i in range(self.num_particles):
            if np.random.rand() < self.mutation_prob:
                mutation_vector = lb + (ub - lb) * np.random.rand(self.dim)
                mutation_index = np.random.randint(0, self.dim)
                self.particles[i][mutation_index] = mutation_vector[mutation_index]
        self.particles = np.clip(self.particles, lb, ub)

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
            self.crossover(lb, ub)
            self.mutate(lb, ub)
        
        return self.global_best_position, self.global_best_fitness