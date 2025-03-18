import numpy as np
from scipy.optimize import minimize

class HGLEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 2 * dim)
        self.current_budget = 0

    def _initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb
        return np.vstack((population, np.flip(population, axis=1)))

    def _evaluate_population(self, population, func):
        evaluations = np.apply_along_axis(func, 1, population)
        self.current_budget += len(population)
        return evaluations

    def _differential_evolution_step(self, population, scores, bounds):
        lb, ub = bounds.lb, bounds.ub
        new_population = []
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = population[idxs]
            mutant = np.clip(a + np.random.rand() * (b - c), lb, ub)  # Adjusted mutation strategy
            cross_points = np.random.rand(self.dim) < 0.9
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, 0.5 * (population[i] + mutant))  # Blended crossover
            new_population.append(trial)
        new_population = np.array(new_population)
        new_scores = self._evaluate_population(new_population, func)
        improved = new_scores < scores
        population[improved] = new_population[improved]
        scores[improved] = new_scores[improved]
        return population, scores

    def _local_optimization(self, best_solution, func, bounds):
        def wrapped_func(x):
            self.current_budget += 1
            return func(x)
        
        result = minimize(wrapped_func, best_solution, bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x if result.success else best_solution

    def __call__(self, func):
        bounds = func.bounds
        population = self._initialize_population(bounds)
        scores = self._evaluate_population(population, func)

        while self.current_budget < self.budget:
            population, scores = self._differential_evolution_step(population, scores, bounds)
            best_idx = np.argmin(scores)
            best_solution = population[best_idx]
            
            if self.current_budget + self.dim <= self.budget:
                best_solution = self._local_optimization(best_solution, func, bounds)
                best_score = func(best_solution)
                self.current_budget += 1
                if best_score < scores[best_idx]:
                    scores[best_idx] = best_score
                    population[best_idx] = best_solution

        best_idx = np.argmin(scores)
        return population[best_idx]