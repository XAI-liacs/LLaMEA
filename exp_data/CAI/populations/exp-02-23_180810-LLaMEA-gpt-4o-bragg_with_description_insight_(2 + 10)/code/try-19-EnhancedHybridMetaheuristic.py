import numpy as np
from scipy.optimize import minimize

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7  # Line changed for adaptive crossover rate
        self.evaluations = 0

    def symmetric_initialization(self, bounds):
        lb, ub = bounds
        mid_point = (ub + lb) / 2
        span = (ub - lb) / 2
        pop = np.random.rand(self.population_size, self.dim) * 2 - 1
        return mid_point + span * pop

    def differential_evolution(self, func, bounds):
        population = self.symmetric_initialization(bounds)
        best_solution = None
        best_score = float('-inf')
        
        # Track best scores for adaptive strategies
        score_history = np.zeros(self.population_size)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation - select the top individuals to improve exploration
                idxs = np.argsort(score_history)[-3:]
                a, b, c = population[idxs]
                mutant = np.clip(a + self.mutation_factor * (b - c), *bounds)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Enforce periodicity to encourage constructive interference
                if np.random.rand() < 0.6:  # Line changed for period enforcement
                    period = np.random.randint(1, self.dim // 2)
                    trial = np.tile(trial[:period], self.dim // period + 1)[:self.dim]

                # Evaluation
                score = func(trial)
                self.evaluations += 1

                # Update score history and selection
                if score > func(population[i]):
                    population[i] = trial
                    score_history[i] = score
                    if score > best_score:
                        best_solution, best_score = trial, score

        return best_solution

    def adaptive_local_search(self, func, best_solution, bounds):
        # Adjust tolerance based on remaining budget
        tol_factor = max(1, self.budget - self.evaluations)
        res = minimize(lambda x: -func(x), best_solution, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-6 / tol_factor})
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        best_solution = self.differential_evolution(func, bounds)
        best_solution = self.adaptive_local_search(func, best_solution, bounds)
        return best_solution