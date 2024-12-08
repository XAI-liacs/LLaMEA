import numpy as np
from scipy.optimize import differential_evolution

class HEAAdHEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.adaptation_rate = 0.05

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set
            updated_population = []
            for i in range(len(elite_set)):
                updated_population.append(elite_set[i])
            for i in range(len(new_population[0:1])):
                updated_population.append(new_population[0:i+1])
                updated_population.append(new_population[0:i])

            elite_set = updated_population[:int(self.budget * self.elitism_ratio)]

            # Apply adaptation rate
            for i in range(len(elite_set)):
                if np.random.rand() < self.adaptation_rate:
                    elite_set[i] = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adhe = HEAAdHEA(budget=100, dim=10)
best_solution = hea_adhe(func)
print(f"Best solution: {best_solution}")