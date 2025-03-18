import numpy as np
from scipy.optimize import minimize

class AdvAPEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        num_subpopulations = 2  # Use multiple subpopulations
        F = 0.8  
        CR = 0.9  
        lb, ub = func.bounds.lb, func.bounds.ub
        
        populations = [lb + (ub - lb) * np.random.rand(population_size, self.dim) for _ in range(num_subpopulations)]
        fitness = [np.array([func(ind) for ind in pop]) for pop in populations]
        eval_count = population_size * num_subpopulations

        cr_min, cr_max = 0.1, 0.9
        CR = [np.full(population_size, CR) for _ in range(num_subpopulations)]
        
        while eval_count < self.budget:
            for s in range(num_subpopulations):  # Iterate over subpopulations
                for i in range(population_size):
                    if eval_count >= self.budget: 
                        break

                    indices = np.random.choice(range(population_size), 3, replace=False)
                    a, b, c = populations[s][indices]
                    F = 0.5 + 0.5 * np.std(fitness[s]) / np.mean(fitness[s])
                    mutant = np.clip(a + F * (b - c), lb, ub)
                    
                    CR[s][i] = np.clip(CR[s][i] + 0.1 * (np.random.rand() - 0.5), cr_min, cr_max)
                    cross_points = np.random.rand(self.dim) < CR[s][i]
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, populations[s][i])
                    
                    trial = self.apply_periodicity(trial, lb, ub)
                    
                    f_trial = func(trial)
                    eval_count += 1

                    if f_trial < fitness[s][i]:
                        populations[s][i] = trial
                        fitness[s][i] = f_trial

            # Local refinement phase
            elite_idx = [np.argmin(fit) for fit in fitness]
            for s in range(num_subpopulations):  # Local search on each subpopulation
                if eval_count + self.dim <= self.budget:
                    bounds = [(lb[j], ub[j]) for j in range(self.dim)]
                    res = minimize(lambda x: func(np.clip(x, lb, ub)), populations[s][elite_idx[s]], method='L-BFGS-B', bounds=bounds)
                    eval_count += res.nfev
                    if res.fun < fitness[s][elite_idx[s]]:
                        populations[s][elite_idx[s]] = res.x
                        fitness[s][elite_idx[s]] = res.fun

        best_subpop_idx = np.argmin([min(fit) for fit in fitness])
        best_idx = np.argmin(fitness[best_subpop_idx])
        return populations[best_subpop_idx][best_idx]

    def apply_periodicity(self, trial, lb, ub):
        period = self.dim // 2
        for i in range(0, self.dim, period):
            period_mean = np.mean(trial[i:i+period])
            trial[i:i+period] = np.clip(period_mean, lb[i:i+period], ub[i:i+period])
        return trial