import numpy as np
from scipy.optimize import minimize

class SynergisticExplorationPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Adjusted population size for better exploration
        self.mutation_factor = 0.85  # Modified mutation factor for improved convergence
        self.crossover_rate = 0.8  # Modified crossover rate for better diversity
        self.evaluations = 0

    def quasi_oppositional_initialization(self, bounds):
        lb, ub = bounds
        mid_point = (ub + lb) / 2
        span = (ub - lb) / 2
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        opposite_population = lb + ub - population
        combined_population = np.vstack((population, opposite_population))
        return combined_population[:self.population_size]

    def adaptive_periodicity(self, trial):
        period = max(1, int(self.dim / (3 + np.log(self.evaluations + 1))))  # Adjusted adaptive periodicity
        return np.tile(trial[:period], self.dim // period + 1)[:self.dim]

    def period_preserving_differential_evolution(self, func, bounds):
        population = self.quasi_oppositional_initialization(bounds)
        best_solution = None
        best_score = float('-inf')

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, *bounds)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                trial = self.adaptive_periodicity(trial)

                score = func(trial)
                self.evaluations += 1

                if score > func(population[i]):
                    population[i] = trial
                    if score > best_score:
                        best_solution, best_score = trial, score

        return best_solution

    def gradient_based_refinement(self, func, best_solution, bounds):
        tol_factor = max(1, self.budget - self.evaluations)
        res = minimize(lambda x: -func(x), best_solution, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-8 / tol_factor})  # Enhanced precision
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        best_solution = self.period_preserving_differential_evolution(func, bounds)
        best_solution = self.gradient_based_refinement(func, best_solution, bounds)
        return best_solution