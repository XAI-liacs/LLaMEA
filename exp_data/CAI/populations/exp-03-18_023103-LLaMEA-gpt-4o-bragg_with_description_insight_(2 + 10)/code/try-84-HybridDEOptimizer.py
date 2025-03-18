import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.8
        self.CR = 0.9
        self.bounds = None
        self.eval_count = 0

    def chaotic_init(self, lb, ub):
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        chaotic_map = 4 * pop * (1 - pop)
        return chaotic_map

    def differential_evolution(self, pop, func):
        new_pop = np.empty_like(pop)
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            indices = np.random.choice(self.pop_size, 3, replace=False)
            x0, x1, x2 = pop[indices]
            self.F = np.random.uniform(0.6, 1.0)
            mutant = np.clip(x0 + self.F * (x1 - x2), self.bounds.lb, self.bounds.ub)
            adaptive_CR = np.random.uniform(0.8, 1.0)
            cross_points = np.random.rand(self.dim) < adaptive_CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = func(trial)
            self.eval_count += 1
            if trial_fitness < func(pop[i]):
                new_pop[i] = self.enforce_periodicity(trial)
            else:
                new_pop[i] = pop[i]
        return new_pop

    def enforce_periodicity(self, solution):
        period = self.dim // 2
        for i in range(0, self.dim, period):
            cosine_weights = (1 + np.cos(2 * np.pi * np.arange(period) / period)) / 2
            solution[i:i+period] = cosine_weights * solution[0:period] + (1 - cosine_weights) * solution[i:i+period]
        return solution

    def local_search(self, best_solution, func):
        if self.eval_count >= self.budget:
            return best_solution
        result = minimize(func, best_solution, bounds=list(zip(self.bounds.lb, self.bounds.ub)), method='L-BFGS-B')
        self.eval_count += result.nfev
        return result.x

    def __call__(self, func):
        self.bounds = func.bounds
        pop = self.chaotic_init(self.bounds.lb, self.bounds.ub)
        best_solution = pop[np.argmin([func(ind) for ind in pop])]
        self.eval_count += self.pop_size

        while self.eval_count < self.budget:
            pop = self.differential_evolution(pop, func)
            current_best = pop[np.argmin([func(ind) for ind in pop])]
            self.eval_count += self.pop_size

            if self.eval_count < self.budget and self.eval_count % (self.pop_size // 2) == 0:
                best_solution = self.local_search(current_best, func)
        
        return best_solution