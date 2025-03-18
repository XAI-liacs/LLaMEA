import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Adaptive population size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8  # Modified differential weight
        self.evals = 0  # Evaluation count
        self.elite_size = max(1, int(0.1 * self.population_size))  # Elitism

    def _initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _mutation(self, target_idx, population, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_f = self.f * (1 - 0.5 * (self.evals / self.budget))  # Updated self-adjusting factor
        mutant = population[a] + adaptive_f * (population[b] - population[c])
        return np.clip(mutant, lb, ub)

    def _crossover(self, target, mutant):
        adaptive_cr = self.cr * (0.5 + 0.5 * (1 - self.evals / self.budget))  # Adaptive crossover probability
        cross_points = np.random.rand(self.dim) < adaptive_cr  # Updated to use adaptive_cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_refinement(self, best, func):
        result = minimize(func, best * 0.985, bounds=zip(func.bounds.lb, func.bounds.ub), method='L-BFGS-B')
        return result.x

    def _adaptive_layer_growth(self, func):
        layer_steps = [10, 20, 32]
        current_dim = min(self.dim, layer_steps[0])
        for next_dim in layer_steps:
            if self.evals > self.budget * next_dim / self.dim:
                current_dim = next_dim
        return current_dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        best_solution = None
        best_score = float('inf')

        while self.evals < self.budget:
            current_dim = self._adaptive_layer_growth(func)
            scores = [func(ind) for ind in population]
            self.evals += len(population)
            elite_indices = np.argsort(scores)[:self.elite_size]
            elite_population = population[elite_indices]

            for i in range(self.population_size):
                target = population[i]
                mutant = self._mutation(i, population, lb, ub)
                trial = self._crossover(target, mutant)
                score = func(trial)
                self.evals += 1
                if score < best_score:
                    best_score = score
                    best_solution = trial

                if score < scores[i]:
                    population[i] = trial

                if self.evals >= self.budget:
                    break

            population = np.vstack((population, elite_population))[:self.population_size]

            if current_dim == self.dim:
                refined_solution = self._local_refinement(best_solution, func)
                refined_score = func(refined_solution)
                self.evals += 1
                if refined_score < best_score:
                    best_score = refined_score
                    best_solution = refined_solution

        return best_solution