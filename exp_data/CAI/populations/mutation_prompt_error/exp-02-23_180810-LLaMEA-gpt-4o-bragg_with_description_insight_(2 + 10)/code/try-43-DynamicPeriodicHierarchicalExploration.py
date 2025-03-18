import numpy as np
from scipy.optimize import minimize

class DynamicPeriodicHierarchicalExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.9
        self.crossover_rate = 0.8
        self.evaluations = 0
        self.max_hierarchy_depth = 3

    def hierarchical_initialization(self, bounds):
        lb, ub = bounds
        span = ub - lb
        population = np.random.rand(self.population_size, self.dim) * span + lb
        for depth in range(1, self.max_hierarchy_depth + 1):
            sub_span = span / (2 ** depth)
            sub_pop = np.random.rand(self.population_size, self.dim) * sub_span + lb
            population = np.vstack((population, sub_pop))
        return population[:self.population_size]

    def dynamic_periodic_adaptive_evolution(self, func, bounds):
        population = self.hierarchical_initialization(bounds)
        best_solution = None
        best_score = float('-inf')

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]

                adaptive_mutation_factor = self.mutation_factor * (1 - (self.evaluations / self.budget)**2)
                mutant = a + adaptive_mutation_factor * (b - c)
                mutant = np.clip(mutant, *bounds)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                period = np.random.randint(1, self.dim // 2 + 1)
                trial = np.tile(trial[:period], self.dim // period + 1)[:self.dim]

                score = func(trial)
                self.evaluations += 1

                if score > func(population[i]):
                    population[i] = trial
                    if score > best_score:
                        best_solution, best_score = trial, score

        return best_solution

    def gradient_refinement(self, func, best_solution, bounds):
        tol_factor = max(1, self.budget - self.evaluations)
        res = minimize(lambda x: -func(x), best_solution, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-6 / tol_factor})
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        best_solution = self.dynamic_periodic_adaptive_evolution(func, bounds)
        best_solution = self.gradient_refinement(func, best_solution, bounds)
        return best_solution