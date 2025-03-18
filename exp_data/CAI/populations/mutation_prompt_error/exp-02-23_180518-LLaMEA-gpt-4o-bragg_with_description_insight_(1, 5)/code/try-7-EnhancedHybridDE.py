import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.7 + np.random.rand() * 0.2  # Crossover probability adjusted for better exploration
        self.pop = None
        self.best_sol = None
        self.best_val = np.inf
        self.evaluations = 0

    def adaptive_quasi_opposition_init(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        midpoint = (lb + ub) / 2
        quasi_opposite_pop = midpoint + (midpoint - self.pop)
        perturbation = np.random.normal(0, 0.1, quasi_opposite_pop.shape)
        quasi_opposite_pop = np.clip(quasi_opposite_pop + perturbation, lb, ub)
        self.pop = np.vstack((self.pop, quasi_opposite_pop))
        self.population_size *= 2

    def promote_periodicity(self, solution):
        periodic_solution = solution.copy()
        for i in range(0, self.dim - 1, 2):
            average_value = (solution[i] + solution[i + 1]) / 2
            periodic_solution[i] = average_value
            periodic_solution[i + 1] = average_value
        return periodic_solution

    def differential_evolution(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.adaptive_quasi_opposition_init(lb, ub)
        fitness = np.apply_along_axis(func, 1, self.pop)
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
                trial = self.promote_periodicity(trial)

                trial_val = func(trial)
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
            result = minimize(func, self.best_sol, method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < self.best_val:
                self.best_val = result.fun
                self.best_sol = result.x

    def __call__(self, func):
        self.differential_evolution(func)
        self.local_search(func)
        return self.best_sol, self.best_val