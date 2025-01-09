import numpy as np

class DynamicMOABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)

        lb = -5.0
        ub = 5.0
        n = 50
        m = 3

        def initialize_population():
            return np.random.uniform(lb, ub, (n, self.dim))

        def update_population(population, fitness_values):
            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]
            return population

        def explore_food_source(x):
            return x + np.random.uniform(-1, 1, x.shape) * (x - population[np.random.randint(n)])

        population = initialize_population()
        fitness_values = np.array([func_wrapper(x) for x in population])

        for _ in range(self.budget):
            for i in range(n):
                new_solution = explore_food_source(population[i])
                new_fitness = func_wrapper(new_solution)
                if new_fitness < fitness_values[i]:
                    population[i] = new_solution
                    fitness_values[i] = new_fitness

            population = update_population(population, fitness_values)

        return population[0]