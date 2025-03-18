import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def periodicity_constraint(self, x):
        """Encourages periodicity by adding a penalty for non-repeating patterns."""
        period_penalty = np.sum((x[:-1] - x[1:])**2)
        return period_penalty

    def __call__(self, func):
        # Initialize parameters for Differential Evolution
        population_size = max(10, self.dim * 5)
        scale_factor = 0.8
        crossover_rate = 0.7
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        best_solution = None
        best_fitness = np.inf
        evals_used = 0

        while evals_used < self.budget:
            new_population = np.zeros_like(population)
            for i in range(population_size):
                if evals_used >= self.budget:
                    break
                # Mutation and crossover
                indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + scale_factor * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution with periodicity penalty
                trial_fitness = func(trial) + self.periodicity_constraint(trial)
                evals_used += 1

                # Replacement
                if trial_fitness < func(population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = population[i]

                # Track the best solution
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

            population = new_population

        # Local refinement with BFGS
        if evals_used < self.budget:
            local_result = minimize(lambda x: func(x) + self.periodicity_constraint(x),
                                    best_solution, method='L-BFGS-B',
                                    bounds=[(func.bounds.lb, func.bounds.ub)] * self.dim,
                                    options={'maxfun': self.budget - evals_used})
            if local_result.fun < best_fitness:
                best_solution = local_result.x

        return best_solution