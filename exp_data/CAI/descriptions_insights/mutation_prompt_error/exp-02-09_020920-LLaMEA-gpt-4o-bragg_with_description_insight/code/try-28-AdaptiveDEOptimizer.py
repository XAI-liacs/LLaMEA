import numpy as np
from scipy.optimize import minimize

class AdaptiveDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8
        CR = 0.9
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        best = None
        best_f = float('inf')
        evaluations = 0
        gen = 0
        historical_diversity = []

        while evaluations < self.budget:
            if evaluations + population_size > self.budget:
                break

            new_population = np.empty_like(population)
            diversity = np.std(population, axis=0)
            historical_diversity.append(diversity)
            avg_diversity = np.mean(historical_diversity, axis=0)

            for i in range(population_size):
                indices = np.arange(population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive Mutation Strategy with Historical Diversity
                adaptive_F = 0.5 + 0.3 * np.random.uniform(-1, 1) * (avg_diversity / (1 + diversity))
                mutant = np.clip(a + adaptive_F * (b - c), lb, ub)

                # Adaptive Crossover Rate
                current_CR = CR * (1 - np.exp(-0.05 * gen))
                cross_points = np.random.rand(self.dim) < current_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial = self.dynamic_periodicity(trial, gen)

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
                # Enhanced Local Search with Strategic Noise
                noise = np.random.uniform(-0.05, 0.05, self.dim) * (1 / (1 + 0.1 * gen))
                opt_result = minimize(func, best + noise, method='L-BFGS-B', bounds=list(zip(lb, ub)))
                evaluations += opt_result.nfev
                if opt_result.fun < best_f:
                    best_f = opt_result.fun
                    best = opt_result.x

        return best

    def dynamic_periodicity(self, solution, generation):
        period = max(2, self.dim // (1 + generation % 3))
        for i in range(0, self.dim, period):
            solution[i:i + period] = np.mean(solution[i:i + period])
        return solution