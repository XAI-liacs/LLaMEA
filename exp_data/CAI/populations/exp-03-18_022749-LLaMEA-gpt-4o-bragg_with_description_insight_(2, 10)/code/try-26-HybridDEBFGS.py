import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.func_evals = 0
        self.periodicity_weight = 0.1

    def differential_evolution(self, func, bounds):
        pop = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        self.func_evals += self.pop_size

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                self.adaptive_mutation_factor()
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
                trial = self.perform_crossover(pop[i], mutant)

                trial_fitness = func(trial)
                self.func_evals += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                
                if self.func_evals >= self.budget:
                    break
                
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

    def adaptive_mutation_factor(self):
        self.mutation_factor = 0.6 * (1 - self.func_evals / self.budget) + 0.4 * (self.func_evals / self.budget)

    def initialize_population(self, bounds):
        pop = np.random.rand(self.pop_size, self.dim)
        return bounds.lb + pop * (bounds.ub - bounds.lb)

    def perform_crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def local_optimization(self, func, x0, bounds):
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.func_evals})
        self.func_evals += res.nfev
        return res.x, res.fun

    def encourage_periodicity(self, x):
        return np.std(x - np.roll(x, shift=1))

    def __call__(self, func):
        def wrapped_func(x):
            return func(x) + self.periodicity_weight * self.encourage_periodicity(x)

        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        bounds_obj = lambda: None
        bounds_obj.lb, bounds_obj.ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        best_sol, best_fitness = self.differential_evolution(wrapped_func, bounds_obj)
        if self.func_evals < self.budget:
            best_sol, best_fitness = self.local_optimization(func, best_sol, bounds)

        return best_sol