import numpy as np

class DEAdaptiveLocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_prob = 0.1  # Probability to do local search

    def differential_evolution(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

            if np.random.rand() < self.local_search_prob:
                self.adaptive_local_search(func, population)
        
        return self.f_opt, self.x_opt

    def adaptive_local_search(self, func, population):
        for _ in range(self.population_size):
            x = population[np.random.randint(self.population_size)]
            step_size = np.random.uniform(0.01, 0.1, size=self.dim) * (func.bounds.ub - func.bounds.lb)
            neighbor = x + np.random.uniform(-1, 1, size=self.dim) * step_size
            neighbor = np.clip(neighbor, func.bounds.lb, func.bounds.ub)
            f = func(neighbor)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = neighbor

    def __call__(self, func):
        return self.differential_evolution(func)