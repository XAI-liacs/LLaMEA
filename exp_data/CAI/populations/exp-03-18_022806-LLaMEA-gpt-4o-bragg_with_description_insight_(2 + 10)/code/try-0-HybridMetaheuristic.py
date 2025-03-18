import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim, periodic_weight=1.0):
        self.budget = budget
        self.dim = dim
        self.periodic_weight = periodic_weight
        self.population_size = 15 * dim  # Recommended for DE
        self.pop = None
        self.fitness = None
        self.bounds = None
        
    def periodic_cost(self, solution, period):
        """Encourage periodicity in the solution."""
        repeated = np.tile(solution[:period], self.dim // period)
        return np.sum((solution - repeated) ** 2)

    def initialize_population(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim)
        self.pop = self.bounds[0] + self.pop * (self.bounds[1] - self.bounds[0])
        self.fitness = np.array([func(ind) for ind in self.pop])
    
    def differential_evolution(self, func):
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover probability
        for generation in range(self.budget // self.population_size):
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, self.pop[i])
                trial_fitness = func(trial) + self.periodic_weight * self.periodic_cost(trial, 2)
                
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.pop[i] = trial

    def local_search(self, func):
        best_idx = np.argmin(self.fitness)
        best_sol = self.pop[best_idx]

        def wrapped_func(x):
            return func(x) + self.periodic_weight * self.periodic_cost(x, 2)

        res = minimize(wrapped_func, best_sol, method='L-BFGS-B', bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
        if res.fun < self.fitness[best_idx]:
            self.pop[best_idx] = res.x
            self.fitness[best_idx] = res.fun

    def __call__(self, func):
        self.initialize_population(func)
        self.differential_evolution(func)
        self.local_search(func)
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]