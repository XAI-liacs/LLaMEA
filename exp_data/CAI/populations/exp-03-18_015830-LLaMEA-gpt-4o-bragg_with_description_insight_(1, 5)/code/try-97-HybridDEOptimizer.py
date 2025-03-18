import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.5
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
        period = max(2, int(self.dim / 4))  # Simplified periodicity calculation
        for i in range(period, self.dim, period):
            individual[i:i+period] = individual[i-period:i]
        return individual

    def enhanced_crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < (self.cr * np.random.rand())  # Dynamically adjusted crossover rate
        cross_points[::2] = True  # Ensure periodicity impact
        return np.where(cross_points, mutant, target)

    def differential_evolution(self, func, lb, ub):
        best_idx = None
        best_val = float('inf')
        success_count = 0

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
                self.f = 0.6 + (0.3 * np.random.rand())  # Modified F-range
                
                mutant = np.clip(a + self.f * (b - c), lb, ub)
                trial = self.enhanced_crossover(self.population[i], mutant)

                trial_val = func(self.add_periodicity(trial))
                self.func_evals += 1
                if trial_val < func(self.population[i]):
                    self.population[i] = trial
                    success_count += 1
                    if trial_val < best_val:
                        best_val = trial_val
                        best_idx = trial
                    self.f = min(1.0, self.f * 1.05)
        return best_idx

    def local_search(self, func, start_point, bounds):
        midpoint = (np.array(bounds)[:,0] + np.array(bounds)[:,1]) / 2
        lb_opposite = midpoint - (start_point - midpoint)
        res = minimize(func, lb_opposite, bounds=bounds, method='L-BFGS-B', options={'maxiter': min(50, self.budget - self.func_evals)})
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