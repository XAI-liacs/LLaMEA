import numpy as np
from scipy.optimize import minimize

class AdvAPEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Initial crossover probability
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize population with symmetric strategy
        population = lb + (ub - lb) * np.random.rand(population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Dynamic crossover adaptation
        cr_min, cr_max = 0.1, 0.9
        CR = np.full(population_size, CR)
        
        while eval_count < self.budget:
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                # Adaptive differential mutation
                indices = np.random.choice(range(population_size), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                # Dynamic crossover adaptation
                CR[i] = np.clip(CR[i] + 0.1 * np.exp(np.random.rand() - 0.5), cr_min, cr_max)
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Enhanced periodicity constraint
                trial = self.enforce_hierarchical_periodicity(trial, lb, ub)
                
                # Calculate fitness
                f_trial = func(trial)
                eval_count += 1

                # Selection and elite preservation
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            # Local refinement phase
            elite_idx = np.argmin(fitness)
            if eval_count + self.dim <= self.budget:
                bounds = [(lb[j], ub[j]) for j in range(self.dim)]
                res = minimize(lambda x: func(np.clip(x, lb, ub)), population[elite_idx], method='L-BFGS-B', bounds=bounds)
                eval_count += res.nfev
                if res.fun < fitness[elite_idx]:
                    population[elite_idx] = res.x
                    fitness[elite_idx] = res.fun

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def enforce_hierarchical_periodicity(self, trial, lb, ub):
        # Force hierarchical periodic patterns in layer thicknesses
        period = self.dim // 2
        for i in range(0, self.dim, period):
            period_mean = np.mean(trial[i:i+period//2])
            trial[i:i+period//2] = np.clip(period_mean, lb[i:i+period//2], ub[i:i+period//2])
            period_mean = np.mean(trial[i+period//2:i+period])
            trial[i+period//2:i+period] = np.clip(period_mean, lb[i+period//2:i+period], ub[i+period//2:i+period])
        return trial