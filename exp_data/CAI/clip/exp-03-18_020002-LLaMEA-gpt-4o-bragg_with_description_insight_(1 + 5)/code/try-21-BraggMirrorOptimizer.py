import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.cr = 0.9
        self.f = 0.8
        self.evaluations = 0
    
    def symmetric_initialization(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        qopop = lb + ub - pop  # Quasi-oppositional solutions
        return np.concatenate((pop, qopop), axis=0)
    
    def differential_evolution_step(self, pop, func, bounds):
        new_pop = np.zeros_like(pop)
        for i in range(len(pop)):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            diversity = np.std(pop, axis=0)
            f_adaptive = self.f * (1 + 0.5 * np.tanh(diversity))
            mutant = np.clip(a + f_adaptive * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr * (1 + 0.7 * np.tanh(diversity.mean()))  # Adjusted crossover
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial = 0.5 * (trial + np.roll(trial, 1)) + 0.1 * np.sin(np.linspace(0, np.pi, self.dim))
            if func(trial) < func(pop[i]):
                new_pop[i] = trial
            else:
                new_pop[i] = pop[i]
            self.evaluations += 1
        return new_pop
    
    def local_optimization(self, best, func, bounds):
        if self.evaluations < 0.8 * self.budget:
            res = minimize(func, best, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
            self.evaluations += res.nfev
            return res.x if res.success else best
        return best
    
    def __call__(self, func):
        bounds = func.bounds
        pop = self.symmetric_initialization(bounds)
        self.pop_size = max(5, self.pop_size - int(self.evaluations / self.budget * 5))  # Dynamic resizing
        while self.evaluations < self.budget:
            pop = self.differential_evolution_step(pop, func, bounds)
            best_idx = np.argmin([func(ind) for ind in pop])
            best = pop[best_idx]
            best = self.local_optimization(best, func, bounds)
            if self.evaluations >= self.budget:
                break
            pop[best_idx] = best
        return best