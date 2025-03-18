import numpy as np
from scipy.optimize import minimize, Bounds

class APWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def periodic_init(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        base_solution = np.random.uniform(lb, ub, self.dim)
        periodic_pop = np.array([(base_solution + i * (ub - lb) / self.dim) % (ub - lb) + lb for i in range(self.dim)])
        return np.clip(periodic_pop, lb, ub)

    def adaptive_differential_evolution(self, population, func, bounds, F=0.5, initial_CR=0.9):
        next_gen = population.copy()
        CR = initial_CR
        for i in range(population.shape[0]):
            indices = list(range(population.shape[0]))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            D = np.random.choice([0.8, 1.0, 1.2])  # Added multi-strategy evolution by using different scaling factors
            F_dynamic = F * (1 - self.evaluations / self.budget)
            mutant = a + D * F_dynamic * (b - c)
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if self.evaluate(func, trial) < self.evaluate(func, population[i]):
                next_gen[i] = trial
                CR = np.clip(CR + np.random.uniform(-0.05, 0.05), 0.1, 1.0)  # Adaptive CR adjustment
            else:
                CR = max(0.1, CR - 0.1)
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
        population = self.periodic_init(bounds)
        best_solution = None
        best_value = np.inf
        
        while self.evaluations < self.budget:
            population = self.adaptive_differential_evolution(population, func, bounds)
            for individual in population:
                global_best = min(population, key=lambda ind: self.evaluate(func, ind))
                local_exploit_weight = self.evaluations / self.budget
                individual = self.local_optimization(func, local_exploit_weight * individual + (1 - local_exploit_weight) * global_best)  # Adaptive exploitation
                value = self.evaluate(func, individual)
                if value < best_value:
                    best_value = value
                    best_solution = individual
        
        return best_solution