import numpy as np

class BioInspired_Random_Walks_Dynamic_Boundary_Adaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.step_size = 0.1
        self.boundary_factor = 0.1
        self.mutation_rate = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_fitness = fitness[best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    new_position = population[i] + np.random.normal(0, self.step_size, self.dim)
                else:
                    new_position = population[i] + self.step_size * (np.random.rand(self.dim) - 0.5)
                
                new_position = np.clip(new_position, lb, ub)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    best_position = new_position
                    best_fitness = new_fitness

                if evaluations >= self.budget:
                    break

            # Dynamic boundary adaptation
            adapt_factor = self.boundary_factor * (1 - evaluations / self.budget)
            lb = np.maximum(func.bounds.lb, lb - adapt_factor * (ub - lb))
            ub = np.minimum(func.bounds.ub, ub + adapt_factor * (ub - lb))

        return best_position, best_fitness