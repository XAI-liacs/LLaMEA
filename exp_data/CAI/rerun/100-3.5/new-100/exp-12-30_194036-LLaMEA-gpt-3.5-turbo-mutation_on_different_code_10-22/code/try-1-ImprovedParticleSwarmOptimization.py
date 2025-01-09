class ImprovedParticleSwarmOptimization(ParticleSwarmOptimization):
    def __init__(self, inertia_min=0.4, inertia_max=0.9, **kwargs):
        super().__init__(**kwargs)
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

    def __call__(self, func):
        for i in range(self.budget):
            inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * i / self.budget
            for i in range(self.n_particles):
                fitness = func(self.particles[i])
                if fitness < self.global_best:
                    self.global_best = fitness
                    self.global_best_pos = self.particles[i].copy()
                
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (self.global_best_pos - self.particles[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (self.global_best_pos - self.particles).mean(axis=0)
                self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]
                
        return self.global_best, self.global_best_pos