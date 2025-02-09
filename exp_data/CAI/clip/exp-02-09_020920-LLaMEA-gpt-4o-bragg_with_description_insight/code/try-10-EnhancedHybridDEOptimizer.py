import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_population_size = 20
        max_population_size = 50
        F = 0.8
        CR = 0.9
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (initial_population_size, self.dim))
        best = None
        best_f = float('inf')

        evaluations = 0
        gen = 0
        while evaluations < self.budget:
            current_population_size = min(max_population_size, initial_population_size + gen)
            if evaluations + current_population_size > self.budget:
                break
            
            new_population = np.empty_like(population)
            for i in range(current_population_size):
                indices = np.arange(current_population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive Mutation Strategy
                F = 0.5 + 0.3 * np.sin(np.pi * gen / 50)
                mutant = np.clip(a + F * (b - c), lb, ub)
                
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
                    new_population[i % initial_population_size] = trial
                else:
                    new_population[i % initial_population_size] = population[i % initial_population_size]
                 
                if f < best_f:
                    best_f = f
                    best = trial

            population = new_population[:initial_population_size]
            gen += 1

            if evaluations < self.budget:
                opt_result = minimize(func, best, method='L-BFGS-B', bounds=list(zip(lb, ub)))
                evaluations += opt_result.nfev
                if opt_result.fun < best_f:
                    best_f = opt_result.fun
                    best = opt_result.x

        return best

    def dynamic_periodicity(self, solution, generation):
        period = max(2, self.dim // (1 + generation % 4))  # Change periodicity dynamic strategy
        for i in range(0, self.dim, period):
            solution[i:i + period] = np.mean(solution[i:i + period])  # Use mean for periodicity
        return solution