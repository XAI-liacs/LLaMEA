import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0

    def periodic_symmetric_initialization(self, bounds):
        lb, ub = bounds
        mid_point = (ub + lb) / 2
        span = (ub - lb) / 2
        pop = np.tile(np.random.rand(self.population_size, self.dim // 2) * 2 - 1, 2)
        return mid_point + span * pop

    def adaptive_mutation_factor(self):
        return 0.5 + np.random.rand() * 0.3

    def differential_evolution(self, func, bounds):
        population = self.periodic_symmetric_initialization(bounds)
        best_solution = None
        best_score = float('-inf')

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.adaptive_mutation_factor() * (b - c), *bounds)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Evaluation
                score = func(trial)
                self.evaluations += 1

                # Selection
                if score > func(population[i]):
                    population[i] = trial
                    if score > best_score:
                        best_solution, best_score = trial, score

        return best_solution

    def local_search(self, func, best_solution, bounds):
        res = minimize(lambda x: -func(x), best_solution, bounds=bounds, method='L-BFGS-B')
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        best_solution = self.differential_evolution(func, bounds)
        best_solution = self.local_search(func, best_solution, bounds)
        return best_solution