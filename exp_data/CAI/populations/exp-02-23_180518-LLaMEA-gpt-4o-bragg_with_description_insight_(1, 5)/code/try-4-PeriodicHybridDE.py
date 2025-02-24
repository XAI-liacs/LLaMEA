import numpy as np
from scipy.optimize import minimize

class PeriodicHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.pop = None
        self.best_sol = None
        self.best_val = np.inf
        self.evaluations = 0

    def quasi_opposition_init(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        midpoint = (lb + ub) / 2
        quasi_opposite_pop = midpoint + (midpoint - self.pop)
        self.pop = np.vstack((self.pop, quasi_opposite_pop))
        self.population_size *= 2

    def periodicity_penalty(self, solution):
        # Calculate penalty for deviations from periodic patterns
        half_dim = self.dim // 2
        penalty = np.sum((solution[:half_dim] - solution[half_dim:]) ** 2)
        return penalty

    def custom_fitness(self, func, solution):
        original_fitness = func(solution)
        penalty = self.periodicity_penalty(solution)
        # Combine original fitness and periodicity penalty
        custom_fitness = original_fitness + penalty
        return custom_fitness

    def differential_evolution(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.quasi_opposition_init(lb, ub)
        fitness = np.apply_along_axis(lambda x: self.custom_fitness(func, x), 1, self.pop)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                trial_val = self.custom_fitness(func, trial)
                self.evaluations += 1

                if trial_val < fitness[i]:
                    fitness[i] = trial_val
                    self.pop[i] = trial

                    if trial_val < self.best_val:
                        self.best_val = trial_val
                        self.best_sol = trial

                if self.evaluations >= self.budget:
                    break

    def local_search(self, func):
        if self.best_sol is not None:
            result = minimize(lambda x: self.custom_fitness(func, x), self.best_sol, method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < self.best_val:
                self.best_val = result.fun
                self.best_sol = result.x

    def __call__(self, func):
        self.differential_evolution(func)
        self.local_search(func)
        return self.best_sol, self.best_val