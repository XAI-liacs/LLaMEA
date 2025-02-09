import numpy as np
from scipy.optimize import minimize

class HybridSymbioticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.subgroup_size = dim // 3  # Reduced subgroup size for diversity
        self.F = 0.9  # More exploration
        self.CR = 0.9  # Increased adaptability
        self.periodicity_enforcement = 0.4  # Enhanced periodicity
        self.adaptive_threshold = 0.8  # More dynamic changes

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim) * 0.95

    def symbiotic_optimization(self, population, bounds, func):
        new_population = np.copy(population)
        for i in range(len(population)):
            indices = [idx for idx in range(len(population)) if idx != i]
            a, b = population[np.random.choice(indices, 2, replace=False)]
            c = np.mean(population, axis=0)  # Use mean for opposition-based learning
            subgroup_indices = np.random.choice(self.dim, self.subgroup_size, replace=False)
            mutant = a.copy()
            mutant[subgroup_indices] = np.clip(a[subgroup_indices] + self.F * (b[subgroup_indices] - c[subgroup_indices]), bounds.lb[subgroup_indices], bounds.ub[subgroup_indices])
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial = self.enforce_periodicity(trial, bounds)

            # Refined symmetry-driven local search
            trial = self.refined_symmetry_search(trial, bounds)

            if func(trial) < func(population[i]):
                new_population[i] = trial

        best_individual = new_population[np.argmin([func(ind) for ind in new_population])]
        new_population[0] = best_individual
        return new_population

    def enforce_periodicity(self, individual, bounds):
        frequency = 2 * np.pi / (bounds.ub - bounds.lb)
        adjusted = individual + self.periodicity_enforcement * np.sin(frequency * (individual - np.mean(individual)))
        return np.clip(adjusted, bounds.lb, bounds.ub)

    def refined_symmetry_search(self, individual, bounds):  # Renamed and adjusted for better clarity
        mirrored = individual[::-1]
        return (individual + mirrored) / 2  # Simplified for better periodicity

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