import numpy as np
from scipy.optimize import minimize, Bounds

class PSE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def periodic_and_random_init(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        base_solution = np.random.uniform(lb, ub, self.dim)
        periodic_part = np.array([(base_solution + i * (ub - lb) / self.dim) % (ub - lb) + lb for i in range(self.dim // 2)])
        random_part = np.random.uniform(lb, ub, (self.dim // 2, self.dim))
        return np.clip(np.vstack((periodic_part, random_part)), lb, ub)

    def symbiotic_evolution(self, population, func, bounds, F=0.5):
        next_gen = population.copy()
        for i in range(population.shape[0]):
            indices = list(range(population.shape[0]))
            indices.remove(i)
            partner = population[np.random.choice(indices)]
            symbiotic_vector = F * (partner - population[i])
            trial = population[i] + symbiotic_vector
            trial = np.clip(trial, bounds.lb, bounds.ub)
            
            if self.evaluate(func, trial) < self.evaluate(func, population[i]):
                next_gen[i] = trial
        
        return next_gen

    def local_optimization(self, func, x):
        bounds = Bounds(func.bounds.lb, func.bounds.ub)
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
        population = self.periodic_and_random_init(bounds)
        best_solution = None
        best_value = np.inf
        
        while self.evaluations < self.budget:
            population = self.symbiotic_evolution(population, func, bounds)
            for individual in population:
                individual = self.local_optimization(func, individual)
                value = self.evaluate(func, individual)
                if value < best_value:
                    best_value = value
                    best_solution = individual
        
        return best_solution