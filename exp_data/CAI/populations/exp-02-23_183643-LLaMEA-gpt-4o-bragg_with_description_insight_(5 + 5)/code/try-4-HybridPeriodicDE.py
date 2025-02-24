import numpy as np
from scipy.optimize import minimize

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim * 5)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        scaled_pop = bounds.lb + pop * (bounds.ub - bounds.lb)
        return scaled_pop

    def _mutate(self, target_idx, population):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), 0, 1)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def _periodic_local_search(self, individual, bounds):
        def objective_wrapper(x):
            return func(x)

        periodic_start = np.mean(individual[::2]), np.mean(individual[1::2])
        initial_guess = np.tile(periodic_start, self.dim // 2)
        result = minimize(objective_wrapper, initial_guess, bounds=bounds, method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = population[i]
                mutant = self._mutate(i, population)
                trial = self._crossover(target, mutant)

                # Scale trial vector to actual bounds
                trial_scaled = bounds.lb + trial * (bounds.ub - bounds.lb)
                trial_fitness = func(trial_scaled)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Periodic-aware local search
            if evaluations < self.budget:
                local_solution, local_fitness = self._periodic_local_search(best_solution, bounds)
                evaluations += 1
                if local_fitness < best_fitness:
                    best_solution = local_solution
                    best_fitness = local_fitness

        return best_solution * (bounds.ub - bounds.lb) + bounds.lb