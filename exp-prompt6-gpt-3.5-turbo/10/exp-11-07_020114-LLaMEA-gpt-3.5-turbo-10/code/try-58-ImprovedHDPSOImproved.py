import numpy as np

class ImprovedHDPSOImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        n_particles = 30
        max_iter = self.budget // n_particles
        c1 = 2.05
        c2 = 2.05
        w_max = 0.9
        w_min = 0.4
        
        def initialize_particles():
            return np.random.uniform(low=-5.0, high=5.0, size=(n_particles, self.dim)), np.zeros((n_particles, self.dim)), np.zeros((n_particles, self.dim))
        
        # Update particle velocity and position with optimized calculations
        def update_particle(particle, velocity, best_particle, best_global):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = 0.5 * velocity + c1 * r1 * (best_particle - particle) + c2 * r2 * (best_global - particle)
            np.clip(particle := particle + velocity, -5.0, 5.0)  # Simultaneously update particle position and clip
            return particle, velocity
        
        # Initialization
        particles, velocities, best_particles = initialize_particles()
        best_global = particles[np.argmin([func(p) for p in particles])]
        
        for _ in range(max_iter):
            for i in range(n_particles):
                particles[i], velocities[i] = update_particle(particles[i], velocities[i], best_particles[i], best_global)
                particle_fitness = func(particles[i])
                best_particle_fitness = func(best_particles[i])
                if particle_fitness < best_particle_fitness:
                    best_particles[i] = particles[i]
                    if particle_fitness < func(best_global):
                        best_global = particles[i]
        
        return best_global