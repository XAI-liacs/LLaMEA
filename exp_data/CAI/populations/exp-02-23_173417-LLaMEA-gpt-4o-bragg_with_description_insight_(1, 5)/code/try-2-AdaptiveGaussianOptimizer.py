import numpy as np
from scipy.optimize import minimize

class AdaptiveGaussianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def adaptive_gaussian_sampling(self, func, bounds, pop_size=20, std_dev=0.1):
        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        while self.evaluations < self.budget:
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break

                # Gaussian sampling around the best solution
                sampled = np.clip(np.random.normal(best_solution, std_dev, self.dim), bounds.lb, bounds.ub)

                # Encourage periodicity
                sampled = self.enforce_periodicity(sampled)

                trial_fitness = func(sampled)
                self.evaluations += 1
                if trial_fitness < fitness[best_idx]:
                    fitness[best_idx] = trial_fitness
                    best_solution = sampled

            # Adaptive adjustment of the standard deviation based on convergence
            std_dev *= 0.95

        return best_solution, fitness[best_idx]

    def enforce_periodicity(self, individual):
        period = 2  # Assuming a basic period of 2 layers for simplicity
        periodic_individual = np.copy(individual)
        for i in range(0, self.dim, period):
            periodic_individual[i:i+period] = np.mean(periodic_individual[i:i+period])  # Average over each period block
        return periodic_individual

    def local_search(self, func, x0, bounds):
        result = minimize(func, x0, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        # Initial global optimization using Adaptive Gaussian Sampling
        best_solution, best_fitness = self.adaptive_gaussian_sampling(func, bounds)

        # Refine the best solution using local search
        best_solution, best_fitness = self.local_search(func, best_solution, bounds)

        return best_solution