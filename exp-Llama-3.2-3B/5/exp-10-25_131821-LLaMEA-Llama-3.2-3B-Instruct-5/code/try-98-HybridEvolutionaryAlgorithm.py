import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.mutation_prob = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])

            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            updated_population = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                if np.random.rand() < self.mutation_prob:
                    updated_population[i] = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
                else:
                    updated_population[i] = new_population[0]

            population = np.concatenate((elite_set, updated_population))

            elite_set = population[:int(self.budget * self.elitism_ratio)]

        best_solution = np.min(func(population))
        return best_solution

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_evolutionary_algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=10)
best_solution = hybrid_evolutionary_algorithm(func)
print(f"Best solution: {best_solution}")