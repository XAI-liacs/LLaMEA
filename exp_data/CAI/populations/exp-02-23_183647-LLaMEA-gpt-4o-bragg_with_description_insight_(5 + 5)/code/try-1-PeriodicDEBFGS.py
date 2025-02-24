import numpy as np
from scipy.optimize import minimize

class PeriodicDEBFGS:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')
        self.bounds = None

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        self.population = self.apply_periodicity(self.population)

        for _ in range(self.budget // self.pop_size):
            new_population = []
            for idx in range(self.pop_size):
                candidates = list(range(0, idx)) + list(range(idx + 1, self.pop_size))
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), *self.bounds)
                mutant = self.apply_periodicity(mutant.reshape(1, -1)).flatten()
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.population[idx])
                trial_score = func(trial)
                
                if trial_score > self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial

                if trial_score > func(self.population[idx]):
                    new_population.append(trial)
                else:
                    new_population.append(self.population[idx])

            self.population = np.array(new_population)
            self.apply_local_search(func)

        return self.best_solution

    def apply_local_search(self, func):
        for individual in self.population:
            result = minimize(lambda x: -func(x), individual, bounds=list(zip(*self.bounds)), method='L-BFGS-B')
            score = -result.fun
            
            if score > self.best_score:
                self.best_score = score
                self.best_solution = result.x

    def apply_periodicity(self, solutions):
        quarter_dim = self.dim // 2
        for solution in solutions:
            solution[:quarter_dim] = solution[quarter_dim:2 * quarter_dim]
            solution[2 * quarter_dim:] = solution[:quarter_dim]
        return solutions