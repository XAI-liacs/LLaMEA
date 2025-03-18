import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def chaotic_initialization(self, lb, ub, size):
        return lb + (ub - lb) * np.sin(np.linspace(0, np.pi, size * self.dim)).reshape(size, self.dim)

    def differential_evolution(self, func, bounds, pop_size=20, F_range=(0.5, 1.0), CR=0.9):
        lb, ub = bounds.lb, bounds.ub
        pop = self.chaotic_initialization(lb, ub, pop_size)
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        
        while self.eval_count < self.budget:
            for i in range(pop_size):
                idxs = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
                x0, x1, x2 = pop[idxs]
                F = np.random.uniform(*F_range)
                mutant = np.clip(x0 + F * (x1 - x2), lb, ub)
                CR = 0.7 + 0.3 * np.random.rand()  # Adaptive crossover rate
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best = trial
                        best_idx = i
                if self.eval_count >= self.budget:
                    break
        return best

    def local_search(self, func, x0, bounds):
        res = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='BFGS')
        self.eval_count += res.nfev
        return res.x

    def __call__(self, func):
        bounds = func.bounds
        best = self.differential_evolution(func, bounds)
        if self.eval_count < self.budget:
            best = self.local_search(func, best, bounds)
        return best