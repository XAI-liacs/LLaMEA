class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.5, phi_p=0.5, phi_g=0.5, inertia_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.inertia_decay = inertia_decay
        self.particles = [Particle(dim, -5.0, 5.0) for _ in range(num_particles)]
        self.global_best = min(self.particles, key=lambda x: func(x.position))
        
    def __call__(self, func):
        for _ in range(self.budget):
            for particle in self.particles:
                new_velocity = self.inertia * particle.velocity + self.phi_p * np.random.rand(self.dim) * (particle.best_position - particle.position) + self.phi_g * np.random.rand(self.dim) * (self.global_best.position - particle.position)
                new_position = particle.position + new_velocity
                new_position = np.clip(new_position, -5.0, 5.0)
                
                f = func(new_position)
                if f < func(particle.best_position):
                    particle.best_position = new_position
                    if f < func(self.global_best.position):
                        self.global_best.position = new_position
            
            self.inertia *= self.inertia_decay
        
        return func(self.global_best.position), self.global_best.position