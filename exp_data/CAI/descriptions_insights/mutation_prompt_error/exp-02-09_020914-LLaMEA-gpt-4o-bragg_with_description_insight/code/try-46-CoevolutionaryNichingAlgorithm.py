import numpy as np
from scipy.optimize import minimize

class CoevolutionaryNichingAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.niches = 5
        self.F = 0.5  # Adaptive adjustment in differential_evolution method
        self.CR = 0.9  # Changed to dynamic adjustment in differential_evolution method
        self.niche_radius = 0.1

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return [lb + (ub - lb) * np.random.rand(self.population_size, self.dim) for _ in range(self.niches)]

    def differential_evolution(self, population, bounds, func):
        new_population = np.copy(population)
        diversity = np.std(population, axis=0).mean()
        F_adaptive = self.F * (0.5 + 0.5 * diversity)  # Adaptive adjustment of F
        best_individual = population[np.argmin([func(ind) for ind in population])]  # Elitism: Keep the best individual
        for i in range(len(population)):
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F_adaptive * (b - c) + self.F * np.random.rand(*a.shape) * (np.mean(population, axis=0) - a), bounds.lb, bounds.ub)  # Diversity-promoting mutation
            CR_dynamic = self.CR * (1.0 - i / len(population))  # Dynamic adjustment of CR
            cross_points = np.random.rand(self.dim) < (0.5 + CR_dynamic / 2)  # Probability-based crossover scheme
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
        new_population[0] = best_individual  # Elitism: Retain the best individual
        return new_population

    def local_optimization(self, x0, func, bounds):
        result = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x if result.success else x0

    def __call__(self, func):
        bounds = func.bounds
        subpopulations = self.initialize_population(bounds)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.niches):
                subpopulations[i] = self.differential_evolution(subpopulations[i], bounds, func)
                evaluations += len(subpopulations[i])
                if evaluations < self.budget:
                    for j in range(len(subpopulations[i])):
                        subpopulations[i][j] = self.local_optimization(subpopulations[i][j], func, bounds)
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

            # Encourage niche behavior
            all_individuals = np.vstack(subpopulations)
            distances = np.linalg.norm(all_individuals[:, np.newaxis, :] - all_individuals[np.newaxis, :, :], axis=2)
            niche_mask = distances < (self.niche_radius * (1 + np.std(all_individuals)))  # Adaptive niche radius adjustment
            niche_fitness = np.array([np.mean([func(ind) for ind in all_individuals[niche_mask[i]]]) for i in range(len(all_individuals))])
            best_niche_idx = np.argmin(niche_fitness)

            if evaluations >= self.budget:
                break

        return all_individuals[best_niche_idx]