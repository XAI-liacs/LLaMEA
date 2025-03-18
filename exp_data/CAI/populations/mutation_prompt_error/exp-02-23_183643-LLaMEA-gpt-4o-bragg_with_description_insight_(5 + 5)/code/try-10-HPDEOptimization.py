import numpy as np
from scipy.optimize import minimize

class HPDEOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = None

    def initialize_population(self, pop_size):
        lb, ub = self.bounds.T  # Corrected to avoid unpacking error by transposing
        return np.random.uniform(lb, ub, (pop_size, self.dim))

    def differential_evolution(self, func, pop, CR=0.9, F=0.8):
        new_pop = np.copy(pop)
        num_individuals = len(pop)
        for i in range(num_individuals):
            indices = [idx for idx in range(num_individuals) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), *self.bounds)
            cross_points = np.random.rand(self.dim) < CR
            trial = np.where(cross_points, mutant, pop[i])
            if func(trial) < func(pop[i]):
                new_pop[i] = trial
        return new_pop

    def local_refinement(self, individual, func):
        result = minimize(func, individual, bounds=self.bounds, method='L-BFGS-B')
        return result.x if result.success else individual

    def __call__(self, func):
        self.bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop_size = 10 * self.dim
        pop = self.initialize_population(pop_size)

        evaluations = 0
        while evaluations < self.budget:
            pop = self.differential_evolution(func, pop)
            if evaluations + pop_size > self.budget:
                break
            evaluations += pop_size

        best = min(pop, key=func)
        if evaluations < self.budget:  # Use remaining budget for refinement
            best = self.local_refinement(best, func)
            evaluations += 1

        return best