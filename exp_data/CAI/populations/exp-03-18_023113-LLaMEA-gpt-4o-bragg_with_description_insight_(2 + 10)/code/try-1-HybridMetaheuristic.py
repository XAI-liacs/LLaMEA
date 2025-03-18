import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def _initialize_population(self, pop_size, lb, ub):
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        opposition_population = lb + ub - population
        combined_population = np.vstack((population, opposition_population))
        return combined_population

    def _evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.eval_count += len(population)
        return fitness

    def _differential_evolution(self, func, pop_size, lb, ub):
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover probability
        population = self._initialize_population(pop_size, lb, ub)
        fitness = self._evaluate_population(population, func)

        while self.eval_count < self.budget:
            for i in range(pop_size):
                indices = np.arange(len(population))
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if self.eval_count >= self.budget:
                break

            # Local search on the best individual found so far
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            local_result = minimize(func, best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')

            if local_result.fun < fitness[best_idx]:
                population[best_idx] = local_result.x
                fitness[best_idx] = local_result.fun

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = 10 * self.dim  # Population size
        best_solution, best_fitness = self._differential_evolution(func, pop_size, lb, ub)
        return best_solution