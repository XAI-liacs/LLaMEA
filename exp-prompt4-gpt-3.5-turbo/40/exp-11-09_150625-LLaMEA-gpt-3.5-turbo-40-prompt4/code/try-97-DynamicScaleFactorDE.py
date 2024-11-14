import numpy as np

class DynamicScaleFactorDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.initial_scale_factor = 0.5
        self.final_scale_factor = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        strategies = np.random.randint(0, 3, self.population_size)  # 0: DE/rand/1, 1: DE/rand/2, 2: DE/best/1

        for gen in range(self.budget - self.population_size):
            current_scale_factor = self.initial_scale_factor - (self.initial_scale_factor - self.final_scale_factor) * gen / (self.budget - self.population_size)
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                if strategies[i] == 0:
                    mutant = population[a] + current_scale_factor * (population[b] - population[c])
                elif strategies[i] == 1:
                    mutant = population[a] + current_scale_factor * (population[b] - population[c]) + current_scale_factor * (population[strategies[(i+1)%self.population_size]] - population[strategies[(i+2)%self.population_size]])
                else:
                    mutant = population[np.argmin(fitness)] + current_scale_factor * (population[a] - population[b])
                
                for j in range(self.dim):
                    if np.random.rand() > self.crossover_rate:
                        mutant[j] = population[i][j]
                mutant_fit = func(mutant)
                if mutant_fit < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fit

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        return best_solution, best_fitness