import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.F = 0.5
        self.CR = 0.9
        self.evaluations = 0
        self.max_layers = dim

    def differential_evolution(self, func, bounds, pop_size=50, F=0.5, CR=0.9):
        pop = np.random.rand(pop_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        best_idx = np.argmin([func(ind) for ind in pop])
        best = pop[best_idx]
        historical_best = best
        while self.evaluations < self.budget:
            diversity = np.mean(np.std(pop, axis=0))
            F = np.clip(0.5 + 0.2 * (diversity / (bounds.ub - bounds.lb).mean()), 0.4, 0.9)
            CR = np.clip(0.7 + 0.2 * (diversity / (bounds.ub - bounds.lb).mean()), 0.6, 1.0)
            adapt_factor = np.random.uniform(0.8, 1.2)
            new_pop_size = int(pop_size * adapt_factor)
            if new_pop_size > pop_size:
                additional_pop = np.random.rand(new_pop_size - pop_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
                pop = np.vstack((pop, additional_pop))
            pop_size = new_pop_size

            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c) + F * (historical_best - b), bounds.lb, bounds.ub)  # Modified Line
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial) + np.var(trial) * 0.005
                self.evaluations += 1
                if f < func(pop[i]):
                    pop[i] = trial
                    if f < func(best):
                        best = trial
                        historical_best = best
                if self.evaluations >= self.budget:
                    break
            if self.evaluations % (self.budget // 6) == 0:
                self.dim = min(self.dim + 3, self.max_layers)
                
        return best

    def local_search(self, func, x0, bounds):
        adaptive_lr = 0.001 + 0.004 * (self.evaluations / self.budget)
        res = minimize(func, x0, bounds=[(low, high) for low, high in zip(bounds.lb, bounds.ub)], method='L-BFGS-B', options={'maxiter': 150, 'learning_rate': adaptive_lr})
        return res.x

    def __call__(self, func):
        bounds = func.bounds
        best = self.differential_evolution(func, bounds, self.pop_size, self.F, self.CR)
        if self.evaluations < self.budget:
            best = self.local_search(func, best, bounds)
        return best