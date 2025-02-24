import numpy as np
from scipy.optimize import minimize

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Initialize bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        self.evals += population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        # Main loop
        while self.evals < self.budget:
            for i in range(population_size):
                # Mutation
                idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_individual = trial

            # Local Search on best individual found
            if self.evals < self.budget:
                res = minimize(
                    func, best_individual, method='L-BFGS-B', bounds=zip(lb, ub),
                    options={'maxiter': min(self.budget - self.evals, 20)}
                )
                self.evals += res.nfev
                if res.fun < fitness[best_idx]:
                    best_individual = res.x
                    fitness[best_idx] = res.fun

            if self.evals >= self.budget:
                break

        return best_individual