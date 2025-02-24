import numpy as np
from scipy.optimize import minimize

class HGLIO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def quasi_oppositional_init(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        mid_point = (lb + ub) / 2
        init_pop = np.random.uniform(lb, ub, (self.dim, self.dim))
        opp_pop = mid_point + (mid_point - init_pop)
        combined_pop = np.vstack((init_pop, opp_pop))
        return np.clip(combined_pop, lb, ub)

    def differential_evolution(self, population, func, bounds, F=0.5, CR=0.9):
        next_gen = population.copy()
        for i in range(population.shape[0]):
            indices = list(range(population.shape[0]))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if self.evaluate(func, trial) < self.evaluate(func, population[i]):
                next_gen[i] = trial
        return next_gen

    def local_optimization(self, func, x):
        bounds = minimize.Bounds(func.bounds.lb, func.bounds.ub)  # Adjusted line
        result = minimize(func, x, method='L-BFGS-B', bounds=bounds)
        return result.x

    def evaluate(self, func, x):
        if self.evaluations < self.budget:
            self.evaluations += 1
            return func(x)
        else:
            return np.inf

    def __call__(self, func):
        bounds = func.bounds
        population = self.quasi_oppositional_init(bounds)
        best_solution = None
        best_value = np.inf
        
        while self.evaluations < self.budget:
            population = self.differential_evolution(population, func, bounds)
            for individual in population:
                individual = self.local_optimization(func, individual)
                value = self.evaluate(func, individual)
                if value < best_value:
                    best_value = value
                    best_solution = individual
        
        return best_solution