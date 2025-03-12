import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.base_F = 0.5  # Base mutation factor
        self.base_CR = 0.9  # Base crossover probability
        self.evaluations = 0
        self.layer_increment = 5  # Increment layers in steps
        self.current_layers = self.layer_increment

    def adaptive_differential_evolution(self, func, bounds, pop_size=50):
        F = self.base_F + np.random.rand() * 0.3  # Adaptive F
        CR = self.base_CR - np.random.rand() * 0.1  # Adaptive CR
        pop = np.random.rand(pop_size, self.current_layers) * (bounds.ub[:self.current_layers] - bounds.lb[:self.current_layers]) + bounds.lb[:self.current_layers]
        best_idx = np.argmin([func(np.pad(ind, (0, self.dim - self.current_layers), 'constant')) for ind in pop])
        best = pop[best_idx]
        while self.evaluations < self.budget:
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb[:self.current_layers], bounds.ub[:self.current_layers])
                cross_points = np.random.rand(self.current_layers) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.current_layers)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(np.pad(trial, (0, self.dim - self.current_layers), 'constant'))
                self.evaluations += 1
                if f < func(np.pad(pop[i], (0, self.dim - self.current_layers), 'constant')):
                    pop[i] = trial
                    if f < func(np.pad(best, (0, self.dim - self.current_layers), 'constant')):
                        best = trial
                if self.evaluations >= self.budget:
                    break
            # Increase complexity gradually
            if self.evaluations % (self.budget // 4) == 0 and self.current_layers < self.dim:
                self.current_layers = min(self.current_layers + self.layer_increment, self.dim)
                pop = np.pad(pop, ((0, 0), (0, self.layer_increment)), 'constant')
        return np.pad(best, (0, self.dim - self.current_layers), 'constant')

    def local_search(self, func, x0, bounds):
        res = minimize(func, x0, bounds=[(low, high) for low, high in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return res.x

    def __call__(self, func):
        bounds = func.bounds
        best = self.adaptive_differential_evolution(func, bounds, self.pop_size)
        if self.evaluations < self.budget:
            best = self.local_search(func, best, bounds)
        return best