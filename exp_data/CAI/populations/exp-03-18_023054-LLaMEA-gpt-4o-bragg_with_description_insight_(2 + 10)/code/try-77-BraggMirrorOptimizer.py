import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def symmetric_initialization(self, lb, ub, size):
        center = (lb + ub) / 2
        return center + (np.random.rand(size, self.dim) - 0.5) * (ub - lb)

    def differential_evolution(self, func, bounds, pop_size=20, F=0.5, CR=0.9):
        lb, ub = bounds.lb, bounds.ub
        pop = self.symmetric_initialization(lb, ub, pop_size)
        best_idx = np.argmin([self.modified_cost(func, ind) for ind in pop])
        best = pop[best_idx].copy()
        
        while self.eval_count < self.budget:
            for i in range(pop_size):
                idxs = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
                x0, x1, x2 = pop[idxs]
                F_adaptive = F * (1 - self.eval_count/self.budget) 
                mutant = np.clip(x0 + F_adaptive * (x1 - x2), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = self.modified_cost(func, trial)
                self.eval_count += 1
                if f_trial < self.modified_cost(func, pop[i]):
                    pop[i] = trial
                    if f_trial < self.modified_cost(func, best):
                        best = trial
                if self.eval_count >= self.budget:
                    break
        return best

    def modified_cost(self, func, solution):
        penalty = np.sum((solution[1:] - solution[:-1])**2)  # Encourage periodicity
        return func(solution) + penalty

    def local_search(self, func, x0, bounds):
        res = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        self.eval_count += res.nfev
        return res.x

    def __call__(self, func):
        bounds = func.bounds
        best = self.differential_evolution(func, bounds)
        if self.eval_count < self.budget:
            best = self.local_search(func, best, bounds)
        return best