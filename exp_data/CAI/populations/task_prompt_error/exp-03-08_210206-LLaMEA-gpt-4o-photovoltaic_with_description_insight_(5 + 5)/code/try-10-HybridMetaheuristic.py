import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.best_solution = None
        self.best_fitness = float('inf')

    def differential_evolution(self, func, bounds):
        bounds = np.array(bounds)  # Corrected line to convert bounds to a numpy array
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.budget -= self.population_size

        while self.budget > 0:
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), bounds[:, 0], bounds[:, 1])
                trial = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])
                trial_fitness = func(trial)
                self.budget -= 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if fitness[i] < self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.best_solution = population[i]

                if self.budget <= 0:
                    break

    def nelder_mead_local_refinement(self, func, bounds):
        if self.best_solution is not None:
            result = minimize(func, self.best_solution, method='Nelder-Mead',
                              bounds=[(low, high) for low, high in zip(bounds[:, 0], bounds[:, 1])],
                              options={'maxiter': self.budget})
            if result.fun < self.best_fitness:
                self.best_fitness = result.fun
                self.best_solution = result.x

    def __call__(self, func):
        bounds = func.bounds
        layers = min(10, self.dim)
        
        while layers <= self.dim:
            sub_bounds = np.array_split(np.column_stack((bounds.lb, bounds.ub)), layers, axis=0)
            def sub_func(x):
                full_x = np.concatenate([x] * (self.dim // layers))[:self.dim]
                return func(full_x)
            
            self.differential_evolution(sub_func, sub_bounds)
            self.nelder_mead_local_refinement(sub_func, sub_bounds)
            layers += 10

        return self.best_solution