import numpy as np
from scipy.optimize import minimize

class BraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def quasi_oppositional_init(self, bounds):
        mid = (bounds.ub + bounds.lb) / 2
        return np.vstack([np.random.uniform(bounds.lb, bounds.ub, self.dim), mid * 2 - np.random.uniform(bounds.lb, bounds.ub, self.dim)])

    def differential_evolution(self, func, bounds):
        pop_size = 10 * self.dim
        population = self.quasi_oppositional_init(bounds)
        while len(population) < pop_size:
            population = np.vstack((population, np.random.uniform(bounds.lb, bounds.ub, self.dim)))

        F_min, F_max = 0.5, 1.0  # Adaptive Differential weight range
        CR_min, CR_max = 0.7, 1.0  # Adaptive Crossover probability range

        best_idx = np.argmin([func(ind) for ind in population])
        best = population[best_idx]

        while self.evaluations < self.budget:
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = population[indices]
                F = np.random.uniform(F_min, F_max)  # Adaptive Differential weight
                CR = np.random.uniform(CR_min, CR_max)  # Adaptive Crossover probability
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                trial_eval = func(trial)
                self.evaluations += 1
                if trial_eval < func(population[i]):
                    population[i] = trial
                    if trial_eval < func(best):
                        best = trial

        return best

    def local_optimization(self, func, x0):
        result = minimize(func, x0, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)])
        return result.x

    def __call__(self, func):
        # Step 1: Global exploration with Differential Evolution
        best_global = self.differential_evolution(func, func.bounds)

        # Step 2: Local refinement with BFGS
        best_local = self.local_optimization(func, best_global)

        return best_local