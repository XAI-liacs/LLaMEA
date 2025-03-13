import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0
        self.max_layers = dim

    def differential_evolution(self, func, bounds, pop_size=50, F=0.5, CR=0.9):
        pop = np.random.rand(pop_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        best_idx = np.argmin([func(ind) for ind in pop])
        best = pop[best_idx]
        while self.evaluations < self.budget:
            diversity = np.mean(np.std(pop, axis=0))
            F = 0.5 + 0.2 * (diversity / (bounds.ub - bounds.lb).mean()) + np.random.normal(0, 0.05)  # Stochastic perturbation
            CR = 0.9 - 0.4 * (diversity / (bounds.ub - bounds.lb).mean()) + np.random.normal(0, 0.02)  # Adaptive control
            if self.evaluations % (self.budget // 10) == 0:
                new_pop_size = min(100, int(pop_size * 1.1))
                if new_pop_size > pop_size:
                    additional_pop = np.random.rand(new_pop_size - pop_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
                    pop = np.vstack((pop, additional_pop))
                pop_size = new_pop_size

            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = np.mean([func(trial) for _ in range(3)]) + np.var(trial) * 0.01
                self.evaluations += 1
                if f < func(pop[i]):
                    pop[i] = trial
                    if f < func(best):
                        best = trial
                if self.evaluations >= self.budget:
                    break
            if self.evaluations % (self.budget // 5) == 0:
                self.dim = min(self.dim + 2, self.max_layers)
        return best

    def local_search(self, func, x0, bounds):
        res = minimize(func, x0, bounds=[(low, high) for low, high in zip(bounds.lb, bounds.ub)], method='L-BFGS-B', options={'maxiter': 100})
        return res.x

    def __call__(self, func):
        bounds = func.bounds
        best = self.differential_evolution(func, bounds, self.pop_size, self.F, self.CR)
        if self.evaluations < self.budget:
            best = self.local_search(func, best, bounds)
        return best