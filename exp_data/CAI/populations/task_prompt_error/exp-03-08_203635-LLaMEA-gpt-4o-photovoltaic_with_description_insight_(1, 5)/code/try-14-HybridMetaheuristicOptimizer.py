import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def adaptive_differential_evolution(self, func, bounds, iters, pop_size=50, F=0.5, CR=0.5):
        population = np.random.rand(pop_size, self.dim)
        population *= (bounds.ub - bounds.lb)
        population += bounds.lb
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        for _ in range(iters):
            if self.evaluations >= self.budget:
                break
            F = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor
            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
        return population[np.argmin(fitness)], np.min(fitness)

    def local_refinement(self, func, bounds, x0):
        # Noise-resilient sampling strategy
        perturbed_solutions = [x0 + np.random.normal(0, 1e-2, self.dim) for _ in range(5)]
        perturbed_fitness = [func(sol) for sol in perturbed_solutions]
        best_perturbed = perturbed_solutions[np.argmin(perturbed_fitness)]
        result = minimize(func, best_perturbed, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        iterations = self.budget // 10
        global_solution, global_cost = self.adaptive_differential_evolution(func, bounds, iters=iterations)
        local_solution, local_cost = self.local_refinement(func, bounds, global_solution)

        if local_cost < global_cost:
            return local_solution
        else:
            return global_solution