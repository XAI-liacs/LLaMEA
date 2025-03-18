import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.5  # Changed from 0.6
        self.cr = 0.92
        self.population = None
        self.func_evals = 0

    def initialize_population(self, lb, ub):
        midpoint = (ub + lb) / 2
        self.population = midpoint + (np.random.rand(self.pop_size, self.dim) - 0.5) * (ub - lb)
        opposite_population = midpoint - (self.population - midpoint)
        self.population = np.vstack((self.population, opposite_population))
        self.func_evals += self.pop_size * 2

    def add_periodicity(self, individual):
        for i in range(1, self.dim, 2):
            individual[i] = individual[i-1]  # Enforce periodicity
        return individual

    def differential_evolution(self, func, lb, ub):
        best_idx = None
        best_val = float('inf')

        for individual in self.population:
            val = func(self.add_periodicity(individual))
            if val < best_val:
                best_val = val
                best_idx = individual

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                # Adaptive mutation factor based on current best value
                adapt_f = self.f + (best_val - 0.8) * 0.1
                mutant = np.clip(a + adapt_f * (b - c), lb, ub)  # Changed mutation strategy
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                trial_val = func(self.add_periodicity(trial))
                self.func_evals += 1
                if trial_val < func(self.population[i]):
                    self.population[i] = trial
                    if trial_val < best_val:
                        best_val = trial_val
                        best_idx = trial

        return best_idx

    def local_search(self, func, start_point, bounds):
        res = minimize(func, start_point, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.func_evals})
        self.func_evals += res.nfev
        return res.x

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))

        self.initialize_population(lb, ub)
        
        best_global = self.differential_evolution(func, lb, ub)
        optimized_solution = self.local_search(func, best_global, bounds)

        return optimized_solution