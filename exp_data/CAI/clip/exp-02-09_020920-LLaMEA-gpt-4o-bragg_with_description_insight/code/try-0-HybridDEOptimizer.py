import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover probability
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        best = None
        best_f = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            if evaluations + population_size > self.budget:
                break  # Ensure we do not exceed budget
            
            new_population = np.empty_like(population)
            for i in range(population_size):
                # Mutation
                indices = np.arange(population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Enforce periodicity: encourage solutions to follow periodic patterns
                trial = self.force_periodicity(trial)

                # Selection
                f = func(trial)
                evaluations += 1
                if f < func(population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = population[i]

                # Update best solution found
                if f < best_f:
                    best_f = f
                    best = trial

            population = new_population

            # Local optimization
            if evaluations < self.budget:
                opt_result = minimize(func, best, method='L-BFGS-B', bounds=[(lb, ub) for _ in range(self.dim)])
                evaluations += opt_result.nfev
                if opt_result.fun < best_f:
                    best_f = opt_result.fun
                    best = opt_result.x

        return best

    def force_periodicity(self, solution):
        period = self.dim // 2
        solution[:period] = solution[period:]
        return solution