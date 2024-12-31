import numpy as np
from scipy.optimize import minimize

class HybridDE_NM:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_threshold = budget // 5

    def differential_evolution(self, func, population):
        for _ in range(self.budget - self.local_search_threshold):
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.bounds[0], self.bounds[1])
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.CR else population[i][j] for j in range(self.dim)])
                f_trial = func(trial_vector)
                f_target = func(population[i])
                if f_trial < f_target:
                    population[i] = trial_vector
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial_vector

    def nelder_mead_local_search(self, func):
        res = minimize(func, self.x_opt, method='Nelder-Mead', bounds=[(self.bounds[0], self.bounds[1])] * self.dim, options={'maxfev': self.local_search_threshold})
        if res.fun < self.f_opt:
            self.f_opt = res.fun
            self.x_opt = res.x

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.differential_evolution(func, population)
        self.nelder_mead_local_search(func)
        return self.f_opt, self.x_opt