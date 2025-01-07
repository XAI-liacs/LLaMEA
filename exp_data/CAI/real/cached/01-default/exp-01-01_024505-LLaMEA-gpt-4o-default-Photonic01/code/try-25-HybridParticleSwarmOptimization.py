import numpy as np

class HybridParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, min(50, budget // 10))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.global_best_position = None
        self.personal_best_fitness = np.full(self.num_particles, float('inf'))
        self.global_best_fitness = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.adaptive_variance = 0.1

    def initialize_particles(self, lb, ub):
        self.positions = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.rand(self.num_particles, self.dim) * (ub - lb) * 0.1
        self.personal_best_positions = np.copy(self.positions)
    
    def evaluate_fitness(self, func):
        fitness = np.array([func(pos) for pos in self.positions])
        for i in range(self.num_particles):
            if fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness[i]
                self.personal_best_positions[i] = self.positions[i]
            if fitness[i] < self.global_best_fitness:
                self.global_best_fitness = fitness[i]
                self.global_best_position = self.positions[i]
        return fitness
    
    def update_velocities_and_positions(self, lb, ub):
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)
        cognitive_component = self.cognitive_weight * r1 * (self.personal_best_positions - self.positions)
        social_component = self.social_weight * r2 * (self.global_best_position - self.positions)
        self.velocities = (self.inertia_weight * self.velocities +
                           cognitive_component +
                           social_component)
        self.positions = np.clip(self.positions + self.velocities, lb, ub)
    
    def apply_quantum_variance(self, lb, ub):
        for i in range(self.num_particles):
            if np.random.rand() < self.adaptive_variance:
                self.positions[i] += np.random.normal(0, self.adaptive_variance, self.dim)
                self.positions[i] = np.clip(self.positions[i], lb, ub)
    
    def adapt_parameters(self, evaluations):
        self.inertia_weight = 0.9 - (0.5 * evaluations / self.budget)
        self.adaptive_variance = 0.1 * (1 - evaluations / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_fitness(func)
            evaluations += len(fitness)

            if evaluations >= self.budget:
                break

            self.update_velocities_and_positions(lb, ub)
            self.apply_quantum_variance(lb, ub)
            self.adapt_parameters(evaluations)
        
        return self.global_best_position, self.global_best_fitness