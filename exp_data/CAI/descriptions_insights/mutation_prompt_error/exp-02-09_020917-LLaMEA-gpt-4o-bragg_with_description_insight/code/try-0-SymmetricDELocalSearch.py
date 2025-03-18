import numpy as np
from scipy.optimize import minimize

class SymmetricDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.lb = None
        self.ub = None

    def symmetric_initialization(self, lb, ub, size):
        midpoint = (lb + ub) / 2
        half_range = (ub - lb) / 2
        init_population = midpoint + np.random.uniform(-half_range, half_range, (size, self.dim))
        return init_population

    def differential_evolution(self, func):
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                f_trial = func(trial)
                f_target = func(self.population[i])
                if f_trial < f_target:
                    self.population[i] = trial

    def local_refinement(self, func):
        best_idx = np.argmin([func(ind) for ind in self.population])
        best_solution = self.population[best_idx]

        result = minimize(func, best_solution, method='BFGS', bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            best_solution = result.x

        return best_solution

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.population = self.symmetric_initialization(self.lb, self.ub, self.population_size)
        self.differential_evolution(func)
        best_solution = self.local_refinement(func)
        return best_solution