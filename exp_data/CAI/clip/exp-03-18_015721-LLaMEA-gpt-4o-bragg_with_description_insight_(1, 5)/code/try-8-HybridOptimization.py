import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def quasi_oppositional_initialization(self, lb, ub, size):
        mid_point = (lb + ub) / 2
        range_span = (ub - lb) / 2
        original = mid_point + np.random.uniform(-range_span, range_span, (size, self.dim))
        opposite = ub - (original - lb)
        choice = np.random.rand(size, self.dim) < 0.5
        return np.where(choice, original, opposite)

    def differential_evolution(self, population, func, bounds):
        CR = 0.9  # Crossover probability
        F = np.random.normal(0.5, 0.1)  # Adjusted mutation factor for exploration
        new_population = np.copy(population)
        for i in range(len(population)):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant_vector, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
                self.evaluations += 1
        return new_population

    def local_search(self, solution, func, bounds):
        if self.evaluations >= self.budget:
            return solution
        res = minimize(func, solution, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        self.evaluations += res.nfev
        return res.x if res.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = 20
        population = self.quasi_oppositional_initialization(bounds.lb, bounds.ub, population_size)  # Updated initialization
        while self.evaluations < self.budget:
            population = self.differential_evolution(population, func, bounds)
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                population[i] = self.local_search(population[i], func, bounds)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]