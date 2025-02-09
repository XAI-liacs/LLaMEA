import numpy as np
from scipy.optimize import minimize

class AdaptiveSymbioticCoEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8
        CR = 0.9
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        best_f = float('inf')
        evaluations = 0
        gen = 0

        while evaluations < self.budget:
            if evaluations + population_size > self.budget:
                break

            new_population = np.empty_like(population)
            for i in range(population_size):
                indices = np.arange(population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Dynamic Periodicity Learning
                F = 0.5 + 0.3 * np.sin(2 * np.pi * gen / 50)
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Adaptive Crossover Rate
                current_CR = CR * (1 - np.exp(-0.05 * gen))
                
                cross_points = np.random.rand(self.dim) < current_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial = self.layer_specific_learning(trial, gen)
                
                f = func(trial)
                evaluations += 1
                if f < func(population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = population[i]

                if f < best_f:
                    best_f = f
                    best = trial

            population = new_population
            gen += 1

            if evaluations < self.budget:
                opt_result = minimize(func, best, method='L-BFGS-B', bounds=list(zip(lb, ub)))
                evaluations += opt_result.nfev
                if opt_result.fun < best_f:
                    best_f = opt_result.fun
                    best = opt_result.x

        return best

    def layer_specific_learning(self, solution, generation):
        period = max(2, self.dim // (1 + generation % 4))  # Changed modulus from 3 to 4
        for i in range(0, self.dim, period):
            segment = solution[i:i + period]
            if np.random.rand() < 0.5:
                segment += np.std(segment) * np.random.randn(*segment.shape)
            else:
                segment -= np.std(segment) * np.random.randn(*segment.shape)
            solution[i:i + period] = segment
        return solution