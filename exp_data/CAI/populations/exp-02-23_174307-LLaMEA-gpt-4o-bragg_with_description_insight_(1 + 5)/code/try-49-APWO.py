import numpy as np
from scipy.optimize import minimize, Bounds

class APWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = 10 + dim  # Adaptive population size

    def periodic_init(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        base_solution = np.random.uniform(lb, ub, self.dim)
        population_size = self.initial_population_size
        periodic_pop = np.array([(base_solution + i * (ub - lb) / self.dim) % (ub - lb) + lb for i in range(population_size)])
        return np.clip(periodic_pop, lb, ub)

    def adaptive_differential_evolution(self, population, func, bounds, F=0.5, initial_CR=0.9):
        next_gen = population.copy()
        CR = initial_CR
        for i in range(population.shape[0]):
            indices = list(range(population.shape[0]))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            F_dynamic = F * (1 - self.evaluations / self.budget)
            mutant = a + F_dynamic * (b - c)
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if self.evaluate(func, trial) < self.evaluate(func, population[i]):
                next_gen[i] = trial
                CR = min(1.0, CR + 0.1)
            else:
                CR = max(0.1, CR - 0.1)
        return next_gen

    def multi_periodic_optimization(self, population, func, bounds):
        multi_pop = []
        for base in population:
            shift = np.random.uniform(-0.1, 0.1, self.dim)  # Slight shift to encourage multi-periodicity
            multi_pop.extend([(base + shift * j) % (bounds.ub - bounds.lb) + bounds.lb for j in range(3)])
        return np.clip(multi_pop, bounds.lb, bounds.ub)

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
        population = self.periodic_init(bounds)
        best_solution = None
        best_value = np.inf
        
        while self.evaluations < self.budget:
            population = self.adaptive_differential_evolution(population, func, bounds)
            if self.evaluations < self.budget / 2:  # Use multi-periodicity for first half 
                population = self.multi_periodic_optimization(population, func, bounds)
            for individual in population:
                individual = self.local_optimization(func, individual)
                value = self.evaluate(func, individual)
                if value < best_value:
                    best_value = value
                    best_solution = individual
        
        return best_solution