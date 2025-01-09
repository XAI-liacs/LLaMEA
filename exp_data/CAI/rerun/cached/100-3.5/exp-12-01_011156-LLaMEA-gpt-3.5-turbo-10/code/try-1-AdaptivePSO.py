import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        def update_velocity_position(particles, global_best):
            inertia_weight = 0.5 + 0.3 * np.cos(2 * np.pi * t / self.budget)  # Dynamic Inertia Weight
            phi_p = 2.05
            phi_g = 2.05
            for particle in particles:
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1, size=self.dim)
                particle.velocity = (inertia_weight * particle.velocity 
                                     + phi_p * r_p * (particle.best_position - particle.position) 
                                     + phi_g * r_g * (global_best - particle.position))
                particle.position = particle.position + particle.velocity
                particle.position = np.clip(particle.position, -5.0, 5.0)
                particle.fitness = func(particle.position)
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = np.copy(particle.position)
                    
            return particles

        class Particle:
            def __init__(self, dim):
                self.position = np.random.uniform(-5.0, 5.0, size=dim)
                self.velocity = np.zeros(dim)
                self.fitness = func(self.position)
                self.best_position = np.copy(self.position)
                self.best_fitness = self.fitness

        max_particles = 30  # Maximum number of particles
        particles = [Particle(self.dim) for _ in range(max_particles)]  # Adaptive particle population
        global_best = min(particles, key=lambda x: x.fitness).best_position
        
        for t in range(self.budget):
            particles = update_velocity_position(particles, global_best)
            global_best = min(particles, key=lambda x: x.fitness).best_position
            
        return global_best