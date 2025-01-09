import numpy as np

class QPSO:
    def __init__(self, budget, dim, num_particles=30, alpha=0.9, beta=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_particle_position(particle, global_best):
            r1, r2 = np.random.rand(), np.random.rand()
            new_position = particle + self.alpha * r1 * (global_best - particle) + self.beta * r2 * (global_best - particle)
            return np.clip(new_position, -5.0, 5.0)

        particles = initialize_particles()
        fitness_values = np.array([func(p) for p in particles])
        global_best = particles[np.argmin(fitness_values)]
        evals = self.num_particles

        while evals < self.budget:
            for i in range(self.num_particles):
                new_position = update_particle_position(particles[i], global_best)
                new_fitness = func(new_position)
                evals += 1

                if new_fitness < fitness_values[i]:
                    particles[i] = new_position
                    fitness_values[i] = new_fitness

                    if new_fitness < func(global_best):
                        global_best = new_position

                if evals >= self.budget:
                    break

        return global_best