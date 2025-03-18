import numpy as np
from scipy.optimize import minimize

class NeighborhoodDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.global_cross_prob = 0.7
        self.local_cross_prob = 0.4
        self.diff_weight = 0.8
        self.local_optimization_prob = 0.1
        self.neighborhood_size = 5  # Number of neighbors considered

    def _initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _differential_evolution_step(self, population, scores, lb, ub, func):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)

            cross_points = np.random.rand(self.dim) < self.global_cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial_score = func(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score

        return population, scores

    def _localized_exploration(self, population, scores, lb, ub, func):
        for i in range(self.population_size):
            neighborhood_indices = np.argsort(scores)[:self.neighborhood_size]
            a, b, c = np.random.choice(neighborhood_indices, 3, replace=False)

            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)

            cross_points = np.random.rand(self.dim) < self.local_cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial_score = func(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score

        return population, scores

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        scores = self._evaluate_population(population, func)

        evaluations = self.population_size
        while evaluations < self.budget:
            population, scores = self._differential_evolution_step(population, scores, lb, ub, func)
            evaluations += self.population_size

            if np.random.rand() < self.local_optimization_prob:
                population, scores = self._localized_exploration(population, scores, lb, ub, func)
                evaluations += self.population_size

        best_idx = np.argmax(scores)
        return population[best_idx]