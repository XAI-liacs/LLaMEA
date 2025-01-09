import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, pop_size=30, c1=2.0, c2=2.0, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def pso_update(particles, velocities, pbest_positions, gbest_position, c1, c2):
            new_velocities = self.f * velocities + c1 * np.random.rand() * (pbest_positions - particles) + c2 * np.random.rand() * (gbest_position - particles)
            return new_velocities

        def de_update(population, f, cr):
            mutant_population = population + f * (population[np.random.permutation(self.pop_size)] - population[np.random.permutation(self.pop_size)])
            crossover_mask = np.random.rand(self.pop_size, self.dim) < cr
            trial_population = np.where(crossover_mask, mutant_population, population)
            return trial_population

        # Initialization
        particles = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest_positions = particles.copy()
        pbest_values = np.array([func(individual) for individual in particles])
        gbest_index = np.argmin(pbest_values)
        gbest_position = pbest_positions[gbest_index]
        gbest_value = pbest_values[gbest_index]

        # Optimization loop
        for _ in range(self.budget):
            new_velocities = pso_update(particles, velocities, pbest_positions, gbest_position, self.c1, self.c2)
            new_particles = particles + new_velocities
            new_particles = np.clip(new_particles, -5.0, 5.0)
            new_values = np.array([func(individual) for individual in new_particles])

            pbest_mask = new_values < pbest_values
            pbest_positions[pbest_mask] = new_particles[pbest_mask]
            pbest_values[pbest_mask] = new_values[pbest_mask]

            gbest_index = np.argmin(pbest_values)
            if pbest_values[gbest_index] < gbest_value:
                gbest_position = pbest_positions[gbest_index]
                gbest_value = pbest_values[gbest_index]

            particles = de_update(particles, self.f, self.cr)

        return gbest_value