import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples
        initial_sample_count = max(10, self.budget // 8)
        
        # Randomly sample initial points within bounds
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        # Evaluate initial samples and find the best one
        best_sample = None
        best_value = float('inf')
        for sample in initial_samples:
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample
            if self.budget <= 0:
                return best_sample

        # Introduce Differential Evolution with adaptive mutation
        population_size = min(20, self.budget // 5)
        population = [best_sample + 0.01 * np.random.randn(self.dim) for _ in range(population_size)]
        
        def differential_evolution(population, best_value):
            F = 0.5 + 0.5 * np.random.rand()  # Adaptive mutation factor
            CR = 0.9  # Crossover rate
            for i in range(population_size):
                a, b, c = np.random.choice(population_size, 3, replace=False)
                mutant = np.clip(population[a] + F * (population[b] - population[c]), func.bounds.lb, func.bounds.ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial_value = func(trial)
                self.budget -= 1
                if trial_value < best_value:
                    best_value = trial_value
                    population[i] = trial
                if self.budget <= 0:
                    break
            return best_value
        
        best_value = differential_evolution(population, best_value)

        # Refine with local optimization
        bounds = [(max(lb, x - 0.15 * (ub - lb)), min(ub, x + 0.15 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        def objective(x):
            return func(x)

        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.8), 'ftol': 1e-7})

        if res.success:
            return res.x
        else:
            return best_sample