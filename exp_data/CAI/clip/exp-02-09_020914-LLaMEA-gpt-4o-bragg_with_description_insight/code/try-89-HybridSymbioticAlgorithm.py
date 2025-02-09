import numpy as np
from scipy.optimize import minimize

class HybridSymbioticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.subgroup_size = dim // 2
        self.F = 0.8  # Adjusted for better exploration
        self.CR = 0.9  # Increased crossover rate for enhanced adaptability
        self.periodicity_enforcement = 0.3  # Reduced to refine periodicity further
        self.adaptive_threshold = 0.75  # Increased to allow dynamic changes

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim) * 0.95

    def symbiotic_optimization(self, population, bounds, func):
        new_population = np.copy(population)
        for i in range(len(population)):
            indices = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            subgroup_indices = np.random.choice(self.dim, self.subgroup_size, replace=False)
            mutant = a.copy()
            mutant[subgroup_indices] = np.clip(a[subgroup_indices] + self.F * (b[subgroup_indices] - c[subgroup_indices]), bounds.lb[subgroup_indices], bounds.ub[subgroup_indices])
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial = self.enforce_periodicity(trial, bounds)

            # Introduce symmetry-driven local search
            trial = self.symmetry_local_search(trial, bounds)

            if func(trial) < func(population[i]):
                new_population[i] = trial

        best_individual = new_population[np.argmin([func(ind) for ind in new_population])]
        new_population[0] = best_individual
        return new_population

    def enforce_periodicity(self, individual, bounds):
        frequency = 2 * np.pi / (bounds.ub - bounds.lb)
        adjusted = individual + self.periodicity_enforcement * np.sin(frequency * individual)
        return np.clip(adjusted, bounds.lb, bounds.ub)

    def symmetry_local_search(self, individual, bounds):
        # Apply symmetry constraints to enhance local search
        mirrored = individual[::-1]
        return (individual + 0.9 * mirrored) / 2  # Adjusted for smoother exploration

    def local_optimization(self, x0, func, bounds):
        result = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x if result.success else x0

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        evaluations = 0

        while evaluations < self.budget:
            population = self.symbiotic_optimization(population, bounds, func)
            evaluations += len(population)

            if evaluations < self.budget:
                for i in range(len(population)):
                    population[i] = self.local_optimization(population[i], func, bounds)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

            if evaluations >= self.budget:
                break

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution