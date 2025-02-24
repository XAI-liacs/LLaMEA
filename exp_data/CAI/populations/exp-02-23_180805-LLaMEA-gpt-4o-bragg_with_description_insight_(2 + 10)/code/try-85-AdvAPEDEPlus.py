import numpy as np
from scipy.optimize import minimize

class AdvAPEDEPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F_min, F_max = 0.5, 0.9
        CR = 0.9
        lb, ub = func.bounds.lb, func.bounds.ub
        
        population = lb + (ub - lb) * np.random.rand(population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size
        
        cr_min, cr_max = 0.1, 0.9
        CR = np.full(population_size, CR)

        while eval_count < self.budget:
            F_dynamic = F_min + np.random.rand() * (F_max - F_min) # Dynamic adaptation of F
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                indices = np.random.choice(range(population_size), 3, replace=False)
                a, b, c = population[indices]
                F = F_dynamic # Use dynamic F instead
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                CR[i] = np.clip(CR[i] + 0.1 * (np.random.rand() - 0.5), cr_min, cr_max)
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial = self.apply_adaptive_periodicity(trial, lb, ub)
                
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            elite_idx = np.argmin(fitness)
            if eval_count + self.dim <= self.budget:
                bounds = [(lb[j], ub[j]) for j in range(self.dim)]
                res = minimize(lambda x: func(np.clip(x, lb, ub)), population[elite_idx], 
                               method='L-BFGS-B', bounds=bounds, options={'maxiter': 20}) # Enhanced local search limit
                eval_count += res.nfev
                if res.fun < fitness[elite_idx]:
                    population[elite_idx] = res.x
                    fitness[elite_idx] = res.fun

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def apply_adaptive_periodicity(self, trial, lb, ub):
        # Adaptive periodic pattern enforcement
        period = self.dim // 2
        for i in range(0, self.dim, period):
            subarray = trial[i:i + period]
            if np.std(subarray) > 0.1:  # Adaptive condition for periodicity
                period_mean = np.mean(subarray)
                trial[i:i + period] = np.clip(period_mean, lb[i:i + period], ub[i:i + period])
        return trial