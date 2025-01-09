class PSO:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.5, phi_p=0.5, phi_g=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_max = inertia
        self.inertia_min = 0.1
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.particles = [Particle(dim, -5.0, 5.0) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(-5.0, 5.0, dim)
        self.global_best_value = np.Inf

    def __call__(self, func):
        for _ in range(self.budget):
            inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
            for particle in self.particles:
                particle.velocity = inertia_weight * particle.velocity + \
                                    self.phi_p * np.random.rand(self.dim) * (particle.personal_best_position - particle.position) + \
                                    self.phi_g * np.random.rand(self.dim) * (self.global_best_position - particle.position)
                particle.position = np.clip(particle.position + particle.velocity, -5.0, 5.0)
                
                f = func(particle.position)
                if f < particle.personal_best_value:
                    particle.personal_best_position = particle.position
                    particle.personal_best_value = f
                
                if f < self.global_best_value:
                    self.global_best_position = particle.position
                    self.global_best_value = f
            
        return self.global_best_value, self.global_best_position