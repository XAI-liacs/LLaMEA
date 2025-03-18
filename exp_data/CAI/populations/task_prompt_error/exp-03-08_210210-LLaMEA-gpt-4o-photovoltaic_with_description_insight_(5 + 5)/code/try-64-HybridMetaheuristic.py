import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 * dim
        self.population_size = self.base_population_size
        self.cr = 0.9
        self.f = 0.5
        self.evals = 0

    def _initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _adaptive_population_size(self):
        self.population_size = max(4, int(self.base_population_size * (1 - self.evals / self.budget)))

    def _dynamic_mutation_factor(self):
        return self.f + (0.2 * (self.evals / self.budget))  # Dynamic mutation factor

    def _mutation(self, target_idx, population, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        dynamic_f = self._dynamic_mutation_factor()
        if np.random.rand() < 0.5:
            mutant = population[a] + dynamic_f * (population[b] - population[c])
        else:
            mutant = population[a] + dynamic_f * (population[b] - population[target_idx])
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
        current_dim = min(self.dim, layer_steps[0])
        for next_dim in layer_steps:
            if self.evals > self.budget * next_dim / self.dim:
                current_dim = next_dim
        return current_dim

    def _tournament_selection(self, population, scores, k=3):
        selected_idx = np.random.choice(self.population_size, k, replace=False)
        best_idx = selected_idx[np.argmin(scores[selected_idx])]
        return population[best_idx]

    def _diversity_preservation(self, population, scores):
        unique_solutions = np.unique(population, axis=0)  # Keep unique solutions
        if len(unique_solutions) < self.population_size:
            extra_needed = self.population_size - len(unique_solutions)
            extra_solutions = self._initialize_population(func.bounds.lb, func.bounds.ub)[:extra_needed]
            population = np.vstack((unique_solutions, extra_solutions))
        return population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        best_solution = None
        best_score = float('inf')
        scores = np.full(self.population_size, np.inf)

        while self.evals < self.budget:
            self._adaptive_population_size()
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

            population = self._diversity_preservation(population, scores)

        return best_solution