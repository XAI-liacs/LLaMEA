import numpy as np

class NovelDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_value = float('inf')
        
    def __call__(self, func):
        function_calls = 0
        fitness = np.apply_along_axis(func, 1, self.population)
        function_calls += self.population_size
        
        while function_calls < self.budget:
            for i in range(self.population_size):
                if function_calls >= self.budget:
                    break

                idxs = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = self.population[idxs]
                diversity = np.std(self.population, axis=0).mean()  # Added line for diversity-based mutation
                mutant = x0 + diversity * (x1 - x2)  # Changed line to use diversity
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, self.population[i])
                
                trial_value = func(trial)
                function_calls += 1
                
                if trial_value < fitness[i]:
                    fitness[i] = trial_value
                    self.population[i] = trial

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial

        return self.best_solution