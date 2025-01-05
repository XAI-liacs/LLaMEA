import numpy as np
from scipy.stats import norm

class HybridDE_BayesianOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0
        self.archive = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            self._evaluate_population(func)

            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                a, b, c = self._select_parents(i)
                F = 0.5 + 0.2 * np.random.rand()
                mutation = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = 0.9 - 0.4 * (self.fitness_evaluations / self.budget)
                trial = self._crossover(self.population[i], mutation, crossover_rate, lower_bound, upper_bound)

                new_fitness = func(trial)
                self.fitness_evaluations += 1
                if new_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = trial

            self._bayesian_refinement(func, lower_bound, upper_bound)

        return self.best_solution

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.fitness_evaluations += 1

            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i]

    def _select_parents(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        selected = np.random.choice(indices, 3, replace=False)
        return self.population[selected[0]], self.population[selected[1]], self.population[selected[2]]

    def _crossover(self, target, mutant, cr, lb, ub):
        mask = np.random.rand(self.dim) < cr
        trial = np.where(mask, mutant, target)
        return np.clip(trial, lb, ub)

    def _bayesian_refinement(self, func, lb, ub):
        if self.fitness_evaluations >= self.budget:
            return

        t = np.arange(self.fitness_evaluations) / self.budget
        mean = np.mean(self.fitness)
        std = np.std(self.fitness) + 1e-6  # Avoid division by zero
        acquisition_values = (self.fitness - mean) / std
        probability_improvement = norm.cdf(acquisition_values)

        selected = np.argmax(probability_improvement)
        refinement_candidate = self.population[selected] + np.random.normal(0, 0.1, self.dim)
        refinement_candidate = np.clip(refinement_candidate, lb, ub)

        candidate_fitness = func(refinement_candidate)
        self.fitness_evaluations += 1

        if candidate_fitness < self.fitness[selected]:
            self.population[selected] = refinement_candidate
            self.fitness[selected] = candidate_fitness
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = refinement_candidate