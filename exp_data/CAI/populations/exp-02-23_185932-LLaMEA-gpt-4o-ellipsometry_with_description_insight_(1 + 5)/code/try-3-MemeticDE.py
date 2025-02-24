import numpy as np
from scipy.optimize import minimize

class MemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population_size = 10
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
        fitness = np.array([self.evaluate_func(ind, func) for ind in population])

        while self.evaluations < self.budget:
            for i in range(population_size):
                # Differential evolution mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = self.evaluate_func(trial, func)

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Local refinement using BFGS on best individual
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            if self.evaluations < self.budget:
                result = minimize(self.evaluate_func, best_individual, args=(func,),
                                  method='L-BFGS-B', bounds=bounds.T,
                                  options={'maxfun': self.budget - self.evaluations})
                population[best_idx] = result.x
                fitness[best_idx] = result.fun

            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            raise RuntimeError("Budget exceeded")