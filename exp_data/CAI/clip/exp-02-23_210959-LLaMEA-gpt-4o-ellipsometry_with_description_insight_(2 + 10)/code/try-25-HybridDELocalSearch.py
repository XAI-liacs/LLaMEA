import numpy as np
from scipy.optimize import minimize

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, int(0.1 * self.budget))  # Define population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
    
    def __call__(self, func):
        # Determine bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget

        # Initialize population
        population = np.random.uniform(lb, ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Differential Evolution loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation: select three distinct individuals
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

        # Local Search: Refine best individual
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            value = func(x)
            evaluations += 1
            return value

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)))

        if result.fun < fitness[best_index]:
            best_solution = result.x

        return best_solution