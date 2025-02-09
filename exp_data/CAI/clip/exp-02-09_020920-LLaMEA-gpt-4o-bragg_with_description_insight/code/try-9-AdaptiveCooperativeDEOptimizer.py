import numpy as np
from scipy.optimize import minimize

class AdaptiveCooperativeDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8
        CR = 0.9
        lb, ub = func.bounds.lb, func.bounds.ub

        # Split population into sub-components for cooperative coevolution
        sub_dim = self.dim // 2
        population1 = np.random.uniform(lb, ub, (population_size, sub_dim))
        population2 = np.random.uniform(lb, ub, (population_size, sub_dim))
        
        best = None
        best_f = float('inf')
        evaluations = 0
        gen = 0

        while evaluations < self.budget:
            if evaluations + population_size > self.budget:
                break
            
            new_population1 = np.empty_like(population1)
            new_population2 = np.empty_like(population2)

            for i in range(population_size):
                indices = np.arange(population_size)
                indices = indices[indices != i]

                # Cooperative coevolution: use different populations
                a1, b1, c1 = population1[np.random.choice(indices, 3, replace=False)]
                a2, b2, c2 = population2[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive mutation strategy for sub-components
                F1 = 0.5 + 0.3 * np.sin(np.pi * gen / 50)
                F2 = 0.5 + 0.3 * np.cos(np.pi * gen / 50)
                
                mutant1 = np.clip(a1 + F1 * (b1 - c1), lb, ub)
                mutant2 = np.clip(a2 + F2 * (b2 - c2), lb, ub)
                
                # Adaptive crossover rate
                current_CR = CR * (1 - np.exp(-0.05 * gen))
                cross_points1 = np.random.rand(sub_dim) < current_CR
                cross_points2 = np.random.rand(sub_dim) < current_CR

                if not np.any(cross_points1):
                    cross_points1[np.random.randint(0, sub_dim)] = True
                if not np.any(cross_points2):
                    cross_points2[np.random.randint(0, sub_dim)] = True

                trial1 = np.where(cross_points1, mutant1, population1[i])
                trial2 = np.where(cross_points2, mutant2, population2[i])

                # Combine trials for evaluation
                trial = np.concatenate((trial1, trial2))
                trial = self.dynamic_periodicity(trial, gen)
                
                # Evaluate combined trial solution
                f = func(trial)
                evaluations += 1
                
                # Update populations based on fitness
                if f < func(np.concatenate((population1[i], population2[i]))):
                    new_population1[i] = trial1
                    new_population2[i] = trial2
                else:
                    new_population1[i] = population1[i]
                    new_population2[i] = population2[i]

                if f < best_f:
                    best_f = f
                    best = trial

            population1, population2 = new_population1, new_population2
            gen += 1

            # Local search using L-BFGS-B
            if evaluations < self.budget:
                opt_result = minimize(func, best, method='L-BFGS-B', bounds=list(zip(lb, ub)))
                evaluations += opt_result.nfev
                if opt_result.fun < best_f:
                    best_f = opt_result.fun
                    best = opt_result.x

        return best

    def dynamic_periodicity(self, solution, generation):
        period = max(2, self.dim // (1 + generation % 2))
        for i in range(0, self.dim, period):
            solution[i:i + period] = np.median(solution[i:i + period])
        return solution