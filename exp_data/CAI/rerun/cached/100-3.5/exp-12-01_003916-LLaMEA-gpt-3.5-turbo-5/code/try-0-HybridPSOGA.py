import numpy as np

class HybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_velocity = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.mutation_rate = 0.1

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim)), np.zeros((self.population_size, self.dim))

        def fitness(x):
            return func(x)

        def PSO():
            swarm, velocity = initialize_population()
            pbest = swarm.copy()
            pbest_fitness = np.apply_along_axis(fitness, 1, swarm)
            gbest = pbest[np.argmin(pbest_fitness)]
            gbest_fitness = np.min(pbest_fitness)

            for _ in range(self.budget):
                r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
                velocity = 0.5 * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
                velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
                swarm += velocity

                current_fitness = np.apply_along_axis(fitness, 1, swarm)
                update_indices = current_fitness < pbest_fitness
                pbest[update_indices] = swarm[update_indices]
                pbest_fitness[update_indices] = current_fitness[update_indices]

                gbest_index = np.argmin(pbest_fitness)
                if pbest_fitness[gbest_index] < gbest_fitness:
                    gbest = pbest[gbest_index]
                    gbest_fitness = pbest_fitness[gbest_index]

            return gbest

        def GA():
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

            for _ in range(self.budget):
                fitness_values = np.apply_along_axis(fitness, 1, population)
                parents = population[np.argsort(fitness_values)[:2]]
                children = parents + self.mutation_rate * np.random.randn(2, self.dim)
                population = np.vstack((population, children))

            best_solution = population[np.argmin(np.apply_along_axis(fitness, 1, population))]
            return best_solution

        return PSO() if np.random.rand() < 0.5 else GA()