import numpy as np

class EnhancedFusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_global_pos = population[np.argmin(fitness)]

        for _ in range(self.budget - self.num_particles):
            r1, r2 = np.random.rand(2, self.num_particles, self.dim)
            velocity_update = self.alpha * velocity + self.beta * r1 * (best_global_pos - population) + self.gamma * r2 * (population - np.mean(population, axis=0))
            new_population = population + velocity_update
            new_fitness = np.array([func(individual) for individual in new_population])
            idx = np.argmin(new_fitness)
            if new_fitness[idx] < fitness.min():
                best_global_pos = new_population[idx]
            population, velocity, fitness = new_population, velocity_update, new_fitness

        return best_global_pos