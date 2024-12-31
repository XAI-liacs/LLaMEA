import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, n_particles=30, inertia=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.bounds = (-5.0, 5.0)
        self.global_best = np.Inf
        self.global_best_pos = None
        self.particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.dim))
        self.velocities = np.zeros((self.n_particles, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            for i in range(self.n_particles):
                fitness = func(self.particles[i])
                if fitness < self.global_best:
                    self.global_best = fitness
                    self.global_best_pos = self.particles[i].copy()
                
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (self.global_best_pos - self.particles[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (self.global_best_pos - self.particles).mean(axis=0)
                self.velocities[i] = self.inertia * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]
                
        return self.global_best, self.global_best_pos