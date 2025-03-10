import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.global_budget_fraction = 0.7  # Fraction of budget for global search
        self.num_layers = 10  # Start with simpler problem

    def __call__(self, func):
        # Prepare DE parameters
        global_budget = int(self.budget * self.global_budget_fraction)
        local_budget = self.budget - global_budget
        pop_size = 10 * self.dim
        F = 0.5  # DE mutation factor
        CR = 0.9  # DE crossover probability

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = pop_size

        # Run DE
        while evaluations < global_budget:
            for i in range(pop_size):
                indices = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                # Include robustness-weighted fitness
                f_trial = func(trial) + 0.01 * np.std(trial)  
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i], fitness[i] = trial, f_trial
            if evaluations >= global_budget:
                break
            F = 0.3 + 0.4 * (evaluations / global_budget)  # Dynamic adjustment of mutation factor

        # Local refinement using Nelder-Mead
        best_idx = np.argmin(fitness)
        result = minimize(func, population[best_idx], method='Nelder-Mead', options={'maxiter': local_budget})
        return result.x

# Example usage:
# optimizer = HybridOptimizer(budget=1000, dim=20)
# best_solution = optimizer(your_func)