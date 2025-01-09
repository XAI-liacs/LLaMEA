import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_weight=0.5, cognitive_weight=1.0, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        
        self.particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best = self.particles.copy()
        
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.num_particles):
                f = func(self.particles[i])
                if f < func(self.personal_best[i]):
                    self.personal_best[i] = self.particles[i].copy()
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = self.particles[i].copy()
                    
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = self.inertia_weight * self.velocities[i] + \
                                     self.cognitive_weight * r1 * (self.personal_best[i] - self.particles[i]) + \
                                     self.social_weight * r2 * (self.x_opt - self.particles[i])
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)
                
        return self.f_opt, self.x_opt