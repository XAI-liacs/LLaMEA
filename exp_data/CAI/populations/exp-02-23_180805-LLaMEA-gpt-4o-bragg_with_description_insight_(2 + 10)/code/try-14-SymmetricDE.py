import numpy as np
from scipy.optimize import minimize

class SymmetricDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize population with symmetric strategy
        population = lb + (ub - lb) * np.random.rand(population_size, self.dim)
        best_idx = np.argmin([func(ind) for ind in population])
        best = population[best_idx].copy()
        eval_count = population_size

        while eval_count < self.budget:
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(range(population_size), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                # Crossover
                CR = np.mean([np.linalg.norm(ind - best) for ind in population]) / (ub - lb).sum()  # Dynamic CR update
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Calculate fitness
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < func(best):
                        best = trial

            # Local refinement using periodic embedding
            if eval_count + self.dim <= self.budget:
                res = minimize(lambda x: func(np.clip(x, lb, ub)), best, 
                               method='L-BFGS-B', 
                               bounds=[(lb[i], ub[i]) for i in range(self.dim)])  # Ensure bounds are tuples
                eval_count += res.nfev
                if res.fun < func(best):
                    best = res.x

        return best