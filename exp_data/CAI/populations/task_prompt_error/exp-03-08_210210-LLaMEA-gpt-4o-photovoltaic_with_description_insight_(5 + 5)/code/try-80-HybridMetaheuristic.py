import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Adaptive population size
        self.cr = 0.9  # Initial crossover probability
        self.f = 0.5  # Differential weight
        self.evals = 0  # Evaluation count

    def _initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _mutation(self, target_idx, population, lb, ub): 
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        # Improved adaptive mutation scaling
        guided_walk = np.random.rand(self.dim) * (population[a] - population[c])
        adaptive_f = self.f + (0.2 * np.random.rand() - 0.1)
        mutant = population[a] + adaptive_f * (population[b] - population[c]) + guided_walk
        return np.clip(mutant, lb, ub)

    def _crossover(self, target, mutant):
        dynamic_cr = self.cr - 0.5 * (self.evals / self.budget)
        cross_points = np.random.rand(self.dim) < dynamic_cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_refinement(self, best, func):
        result = minimize(func, best, bounds=zip(func.bounds.lb, func.bounds.ub), method='L-BFGS-B')
        return result.x

    def _adaptive_layer_growth(self, func):
        layer_steps = [10, 20, 32]
        current_dim = layer_steps[0]
        for next_dim in layer_steps:
            if self.evals > self.budget * next_dim / self.dim:
                current_dim = next_dim
        current_dim = min(current_dim + (self.evals // (self.budget // len(layer_steps))), self.dim)
        return current_dim

    def _tournament_selection(self, population, scores, k=3):
        selected_idx = np.random.choice(self.population_size, k, replace=False)
        best_idx = selected_idx[np.argmin(scores[selected_idx])]
        return population[best_idx]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        best_solution = None
        best_score = float('inf')
        scores = np.full(self.population_size, np.inf)

        while self.evals < self.budget:
            current_dim = self._adaptive_layer_growth(func)
            for i in range(self.population_size):
                target = self._tournament_selection(population, scores)
                mutant = self._mutation(i, population, lb, ub)
                trial = self._crossover(target, mutant)
                score = func(trial)
                self.evals += 1
                scores[i] = score
                if score < best_score:
                    best_score = score
                    best_solution = trial

                if score < scores[i]:
                    population[i] = trial

                if self.evals >= self.budget:
                    break

            if current_dim == self.dim:
                refined_solution = self._local_refinement(best_solution, func)
                refined_score = func(refined_solution)
                self.evals += 1
                if refined_score < best_score:
                    best_score = refined_score
                    best_solution = refined_solution

        return best_solution