import numpy as np

class BIAGO:
    def __init__(self, budget, dim, population_size=30, diffusion_rate=0.1, learning_rate=0.01, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.diffusion_rate = diffusion_rate
        self.learning_rate = learning_rate
        self.mutation_prob = mutation_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_position = None
        best_value = float('inf')

        population = self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Apply stochastic diffusion
                diffused_position = self.stochastic_diffusion(population[i], lb, ub)

                # Evaluate the function at the new position
                value = func(diffused_position)
                self.evaluations += 1

                if value < best_value:
                    best_value = value
                    best_position = diffused_position

                if self.evaluations >= self.budget:
                    break

            # Evolve population using evolutionary learning
            self.evolve_population(population, lb, ub)

            if self.evaluations >= self.budget:
                break

        return best_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def stochastic_diffusion(self, position, lb, ub):
        noise = np.random.normal(0, self.diffusion_rate, self.dim)
        new_position = position + noise
        return np.clip(new_position, lb, ub)

    def evolve_population(self, population, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_prob:
                mutation = (np.random.rand(self.dim) - 0.5) * self.learning_rate * (ub - lb)
                population[i] = np.clip(population[i] + mutation, lb, ub)

        # Selection step: keep only the best half of the population
        half_size = self.population_size // 2
        fitness = np.array([func(ind) for ind in population])
        best_indices = np.argsort(fitness)[:half_size]
        population[:half_size] = population[best_indices]
        
        # Refill the population by cloning best solutions
        for i in range(half_size, self.population_size):
            parent = population[np.random.choice(half_size)]
            population[i] = self.stochastic_diffusion(parent, lb, ub)