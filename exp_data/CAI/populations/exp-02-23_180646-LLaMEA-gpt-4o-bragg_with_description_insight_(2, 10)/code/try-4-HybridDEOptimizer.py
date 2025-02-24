import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = None
        self.evaluations = 0

    def quasi_oppositional_initialization(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        x_quasi_opposite = lb + ub - pop
        combined = np.vstack((pop, x_quasi_opposite))
        return combined

    def differential_evolution(self, func):
        CR = 0.9
        population = self.quasi_oppositional_initialization(self.bounds.lb, self.bounds.ub)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += population.shape[0]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = np.random.choice(len(population), 3, replace=False)
                a, b, c = population[indices]
                F = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                self.evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def local_search(self, func, initial_solution):
        result = minimize(func, initial_solution, method="L-BFGS-B", bounds=np.vstack((self.bounds.lb, self.bounds.ub)).T)
        return result.x, result.fun

    def __call__(self, func):
        self.bounds = func.bounds
        best_solution, best_fitness = self.differential_evolution(func)

        if self.evaluations < self.budget:
            best_solution, best_fitness = self.local_search(func, best_solution)

        return best_solution