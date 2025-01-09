import numpy as np

class PSO:
    def __init__(self, budget, dim, num_particles=30, max_velocity=0.2):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_velocity = max_velocity
        self.inertia_weight = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.global_best = None

    def __call__(self, func):
        def fitness(particle):
            return func(particle)
        
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim)), \
                   np.zeros((self.num_particles, self.dim)), \
                   np.full(self.num_particles, np.inf), \
                   np.random.uniform(-5.0, 5.0, size=self.dim)
        
        particles, velocities, personal_bests, global_best = initialize_particles()
        
        for _ in range(self.budget):
            for i in range(self.num_particles):
                current_fitness = fitness(particles[i])
                
                if current_fitness < personal_bests[i]:
                    personal_bests[i] = current_fitness
                    if current_fitness < personal_bests[i]:
                        self.global_best = particles[i].copy()
                
                velocities[i] = self.inertia_weight * velocities[i] + \
                                self.c1 * np.random.rand(self.dim) * (particles[i] - particles[i]) + \
                                self.c2 * np.random.rand(self.dim) * (self.global_best - particles[i])
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
        
        return self.global_best