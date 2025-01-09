import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = np.inf

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = func.bounds
        particles = [Particle(self.dim, bounds) for _ in range(self.num_particles)]
        global_best_value = np.inf
        global_best_position = None
        
        w = 0.5  # inertia weight
        c1 = 1.5 # cognitive component
        c2 = 1.5 # social component

        evals = 0
        while evals < self.budget:
            for particle in particles:
                f_value = func(particle.position)
                evals += 1
                if f_value < particle.best_value:
                    particle.best_value = f_value
                    particle.best_position = np.copy(particle.position)

                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = np.copy(particle.position)

                if evals >= self.budget:
                    break

            for particle in particles:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
                social_velocity = c2 * r2 * (global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity
                particle.position += particle.velocity

                # Ensure the particle remains in the bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt