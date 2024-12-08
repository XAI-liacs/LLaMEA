import random
import numpy as np

class HyperElitist:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elitist = None
        self.population = self.initialize_population()

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        # Ensure that the first 10% of the population are unique
        unique_indices = random.sample(range(self.budget), int(self.budget * 0.1))
        for i in unique_indices:
            population[i] = np.random.uniform(-5.0, 5.0, self.dim)
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            # Select the best 20% of the population
            selected = self.population[np.argsort(func(self.population))[:int(0.2 * self.budget)]]
            # Select the best individual from the selected population using Bloom filter
            bloom_filter = np.zeros(self.budget, dtype=bool)
            for individual in selected:
                np.random.seed(individual[0])
                bloom_filter[int(np.random.uniform(0, self.budget))]=True
            best_indices = np.where(bloom_filter)[0]
            best = selected[best_indices[np.argmin(func(selected[best_indices]])]]
            # Add the best individual to the elitist list
            if self.elitist is None or np.linalg.norm(best - self.elitist) > 1e-6:
                self.elitist = best
            # Replace the worst individual with the new best individual
            self.population[self.budget - 1] = best
            # Replace the worst individual in the selected population with a new random individual
            worst = selected[np.argmax(func(selected))]
            self.population[np.argmin(func(self.population))] = np.random.uniform(-5.0, 5.0, self.dim)

    def get_elitist(self):
        return self.elitist

# Example usage:
def func(x):
    return np.sum(x**2)

hyper_elitist = HyperElitist(100, 10)
hyper_elitist(func)
print(hyper_elitist.get_elitist())