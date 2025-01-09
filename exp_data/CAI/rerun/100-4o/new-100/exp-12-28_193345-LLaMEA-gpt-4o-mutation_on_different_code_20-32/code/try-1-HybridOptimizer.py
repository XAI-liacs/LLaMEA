import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def differential_evolution(self, func, pop_size=20, F=0.5, CR=0.9):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += pop_size

        adapt_factor = 0.1  # New: adaptive parameter
        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + (F + adapt_factor * np.random.rand()) * (b - c), bounds[0], bounds[1])  # Modified
                crossover = np.random.rand(self.dim) < CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        adapt_factor *= 0.9  # Adaptive adjustment

        return population, fitness

    def nelder_mead(self, func, x0):
        result = minimize(func, x0, method='Nelder-Mead', options={'maxfev': self.budget - self.eval_count})
        self.eval_count += result.nfev
        if result.fun < self.f_opt:
            self.f_opt = result.fun
            self.x_opt = result.x

    def __call__(self, func):
        pop, fitness = self.differential_evolution(func)
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx]
        self.nelder_mead(func, best_sol)
        return self.f_opt, self.x_opt