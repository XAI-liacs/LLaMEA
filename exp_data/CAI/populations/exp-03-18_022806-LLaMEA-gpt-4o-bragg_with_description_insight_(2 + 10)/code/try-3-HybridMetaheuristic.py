import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim, periodic_weight=1.0):
        self.budget = budget
        self.dim = dim
        self.periodic_weight = periodic_weight
        self.population_size = 15 * dim
        self.pop = None
        self.fitness = None
        self.bounds = None
        
    def adaptive_periodic_cost(self, solution):
        """Adaptively encourage periodicity in the solution."""
        min_period = 2
        max_period = min(10, self.dim // 2)
        cost = np.inf
        for period in range(min_period, max_period + 1):
            repeated = np.tile(solution[:period], self.dim // period)
            current_cost = np.sum((solution - repeated) ** 2)
            cost = min(cost, current_cost)
        return cost

    def chaotically_initialize_population(self, func):
        """Use chaotic initialization to diversify the population."""
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.pop = np.random.rand(self.population_size, self.dim)
        self.pop = np.sin(self.pop * np.pi) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.fitness = np.array([func(ind) for ind in self.pop])
    
    def differential_evolution(self, func):
        F = 0.8
        CR = 0.9
        for generation in range(self.budget // self.population_size):
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, self.pop[i])
                trial_fitness = func(trial) + self.periodic_weight * self.adaptive_periodic_cost(trial)
                
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.pop[i] = trial

    def local_search(self, func):
        best_idx = np.argmin(self.fitness)
        best_sol = self.pop[best_idx]

        def wrapped_func(x):
            return func(x) + self.periodic_weight * self.adaptive_periodic_cost(x)

        res = minimize(wrapped_func, best_sol, method='L-BFGS-B', bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
        if res.fun < self.fitness[best_idx]:
            self.pop[best_idx] = res.x
            self.fitness[best_idx] = res.fun

    def __call__(self, func):
        self.chaotically_initialize_population(func)
        self.differential_evolution(func)
        self.local_search(func)
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx]