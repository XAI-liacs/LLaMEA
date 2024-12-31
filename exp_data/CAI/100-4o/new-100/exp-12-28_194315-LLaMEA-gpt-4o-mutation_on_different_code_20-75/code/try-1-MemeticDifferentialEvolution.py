import numpy as np
from scipy.optimize import minimize

class MemeticDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def differential_evolution(self, pop, func):
        for i in range(self.pop_size):
            indices = list(range(self.pop_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            self.F = 0.5 + np.random.rand() * 0.5  # Adaptive F
            mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            f = func(trial)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial
            if f < func(pop[i]):
                pop[i] = trial
        return pop

    def local_search(self, x, func):
        if np.random.rand() < 0.3:  # Randomly select candidates for local search
            res = minimize(func, x, method='Nelder-Mead', options={'maxiter': self.budget // 10, 'disp': False})
            if res.fun < self.f_opt:
                self.f_opt = res.fun
                self.x_opt = res.x

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        func_calls = 0

        while func_calls < self.budget:
            pop = self.differential_evolution(pop, func)
            func_calls += self.pop_size
            for candidate in pop:
                if func_calls < self.budget:
                    self.local_search(candidate, func)
                    func_calls += self.budget // 10
                else:
                    break

        return self.f_opt, self.x_opt