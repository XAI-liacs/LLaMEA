import numpy as np
import random

class HybridHSPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.probability = 0.25

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of each individual in the population
            fitness = func(self.x)

            # Update the pbest and gbest
            for i in range(self.population_size):
                if np.any(fitness[i] < fitness[self.pbest[i, :]]):
                    self.pbest[i, :] = self.x[i, :]
                if np.any(fitness[i] < fitness[self.gbest]):
                    self.gbest = self.x[i, :]

            # Select the fittest individuals for crossover and mutation
            fittest_indices = np.argsort(fitness)[:, ::-1][:, :int(self.population_size * self.crossover_rate)]
            fittest_x = self.x[fittest_indices]

            # Perform crossover and mutation
            new_x = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    new_x[i, :] = fittest_x[np.random.randint(0, len(fittest_x)), :]
                else:
                    new_x[i, :] = self.x[i, :]
                if np.random.rand() < self.mutation_rate:
                    new_x[i, :] += np.random.uniform(-0.1, 0.1, self.dim)

            # Refine the strategy with probability 0.25
            if np.random.rand() < self.probability:
                new_x = self.refine_strategy(new_x, fittest_x, fitness)

            # Ensure the search space bounds
            new_x = np.clip(new_x, -5.0, 5.0, out=new_x)

            # Replace the old population with the new one
            self.x = new_x

            # Evaluate the fitness of the new population
            fitness = func(self.x)

            # Update the pbest and gbest
            for i in range(self.population_size):
                if np.any(fitness[i] < fitness[self.pbest[i, :]]):
                    self.pbest[i, :] = self.x[i, :]
                if np.any(fitness[i] < fitness[self.gbest]):
                    self.gbest = self.x[i, :]

            # Check if the optimization is complete
            if np.all(fitness < 1e-6):
                break

    def refine_strategy(self, new_x, fittest_x, fitness):
        # Select 25% of the fittest individuals
        selected_indices = np.random.choice(len(fittest_x), size=int(len(fittest_x) * self.probability), replace=False)
        selected_x = fittest_x[selected_indices]

        # Create a new individual by combining the selected individuals
        new_individual = np.zeros_like(new_x)
        for i in range(self.population_size):
            new_individual[i, :] = new_x[i, :] + np.random.uniform(-0.1, 0.1, self.dim)

            # Replace the individual with a random selection from the selected individuals
            if np.random.rand() < 0.5:
                new_individual[i, :] = selected_x[np.random.randint(0, len(selected_x)), :]

        return new_individual

# Example usage
def noiseless_func(x):
    return np.sum(x**2)

hybridHSPSO = HybridHSPSO(budget=100, dim=10)
hybridHSPSO noiseless_func