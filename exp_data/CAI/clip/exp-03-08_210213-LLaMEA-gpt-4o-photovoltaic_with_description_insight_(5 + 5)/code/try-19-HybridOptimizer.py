import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.global_budget_fraction = 0.7  # Fraction of budget for global search
        self.num_layers = 10  # Start with simpler problem

    def __call__(self, func):
        global_budget = int(self.budget * self.global_budget_fraction)
        local_budget = self.budget - global_budget
        pop_size = 10 * self.dim

        # Self-adaptive DE parameters
        F = np.random.rand(pop_size) * 0.5 + 0.5  # Mutate between 0.5 and 1.0
        CR = np.random.rand(pop_size) * 0.6 + 0.3  # Crossover between 0.3 and 0.9

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = pop_size

        while evaluations < global_budget:
            for i in range(pop_size):
                indices = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i], fitness[i] = trial, f_trial

            # Apply robustness perturbation check
            if evaluations >= global_budget / 2:
                perturbation = np.random.normal(scale=0.05, size=(pop_size, self.dim))
                fitness_perturbed = np.array([func(ind + perturbation[i]) for i, ind in enumerate(population)])
                fitness = np.minimum(fitness, fitness_perturbed)
            if evaluations >= global_budget:
                break

        best_idx = np.argmin(fitness)
        result = minimize(func, population[best_idx], method='Nelder-Mead', options={'maxiter': local_budget})
        return result.x