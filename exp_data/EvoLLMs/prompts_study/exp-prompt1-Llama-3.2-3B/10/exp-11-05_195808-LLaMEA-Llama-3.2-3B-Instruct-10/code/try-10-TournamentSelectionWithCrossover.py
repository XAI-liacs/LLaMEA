import numpy as np
import random

class TournamentSelectionWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_init_method = 'random'

    def __call__(self, func):
        # Initialize population with random points
        if self.population_init_method == 'random':
            population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        elif self.population_init_method =='spherical':
            population = np.random.normal(0, 1, size=(self.budget, self.dim))
            population = np.clip(population, self.lower_bound, self.upper_bound)

        # Evaluate population
        fitness = np.array([func(point) for point in population])

        # Selection
        for _ in range(self.budget):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            # Get best point
            best_point = tournament_points[np.argmin(tournament_fitness)]

            # Crossover
            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            # Mutation
            mutation_rate = 0.1
            if np.random.rand() < mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Replace worst point
            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

        # Maintain diversity
        diversity = np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))
        if diversity < 0.5:
            # Replace worst point with a new random point
            new_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            population[worst_index] = new_point
            fitness[worst_index] = func(new_point)

        return population[0], fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)