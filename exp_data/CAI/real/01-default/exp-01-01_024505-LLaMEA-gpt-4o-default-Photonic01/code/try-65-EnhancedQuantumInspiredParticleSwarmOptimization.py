import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimization:
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
        self.inertia_weight = 0.9  # Increase initial inertia for wider exploration
        self.constriction_factor = 0.5
        self.dynamic_neighborhood_size = 5  # Larger neighborhood to improve local exploration
        self.interference_prob = 0.1
        self.diversity_threshold = 0.1  # Diversity threshold for reinitialization
        self.adaptive_rate = 0.995  # Adaptive decay rate for inertia

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
            local_best = self.get_local_best(i)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_term = self.cognitive_const * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_term = self.social_const * r2 * (local_best - self.particles[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_term + social_term)
            self.particles[i] += self.constriction_factor * self.velocities[i]
        self.particles = np.clip(self.particles, lb, ub)
        self.inertia_weight *= self.adaptive_rate  # Decay inertia for better convergence

    def get_local_best(self, index):
        neighborhood_indices = np.random.choice(self.num_particles, self.dynamic_neighborhood_size, replace=False)
        local_best_index = min(neighborhood_indices, key=lambda i: self.personal_best_fitness[i])
        return self.personal_best_positions[local_best_index]

    def apply_quantum_interference(self, lb, ub):
        for i in range(self.num_particles):
            if np.random.rand() < self.interference_prob:
                interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.particles[i] = np.mean([self.particles[i], interference_vector], axis=0)
                self.particles[i] = np.clip(self.particles[i], lb, ub)

    def preserve_diversity(self, lb, ub):
        diversity = np.std(self.particles, axis=0)
        if np.any(diversity < self.diversity_threshold):
            reinit_indices = np.random.choice(self.num_particles, size=self.num_particles // 2, replace=False)
            self.particles[reinit_indices] = lb + (ub - lb) * np.random.rand(len(reinit_indices), self.dim)
            self.velocities[reinit_indices] = np.random.randn(len(reinit_indices), self.dim) * 0.1

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
            self.apply_quantum_interference(lb, ub)
            self.preserve_diversity(lb, ub)

        return self.global_best_position, self.global_best_fitness