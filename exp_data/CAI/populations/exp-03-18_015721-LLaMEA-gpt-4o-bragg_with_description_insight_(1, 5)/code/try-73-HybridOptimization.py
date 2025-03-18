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
        CR = 0.9 - 0.6 * (self.evaluations / self.budget)  # Adjusted crossover probability
        F = 0.4 + np.random.rand() * 0.6  # Expanded adaptive Differential weight range
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
            trial = self.apply_periodicity(trial, bounds)
            if func(trial) < func(population[i]):
                new_population[i] = trial
                self.evaluations += 1
        return new_population

    def apply_periodicity(self, solution, bounds):
        period_length = max(1, self.dim // (6 + int(self.evaluations / self.budget * 12)))  # More aggressive period adjustment
        for i in range(self.dim):
            solution[i] = solution[i % period_length]
        return np.clip(solution, bounds.lb, bounds.ub)

    def local_search(self, solution, func, bounds):
        if self.evaluations >= self.budget or np.random.rand() > (1 - self.evaluations / self.budget):
            return solution
        max_iter = 5 + int(self.evaluations / self.budget * 15)  # Multi-fidelity local search iterations
        res = minimize(func, solution, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B', options={'maxiter': max_iter})
        self.evaluations += res.nfev
        return res.x if res.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = 25  # Increased population size for better exploration
        population = self.quasi_oppositional_initialization(bounds.lb, bounds.ub, population_size)
        while self.evaluations < self.budget:
            population = self.differential_evolution(population, func, bounds)
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                population[i] = self.local_search(population[i], func, bounds)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]