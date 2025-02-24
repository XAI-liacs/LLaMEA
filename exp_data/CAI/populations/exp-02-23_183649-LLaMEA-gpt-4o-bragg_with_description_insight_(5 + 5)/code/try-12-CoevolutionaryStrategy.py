import numpy as np
import scipy.optimize as opt

class CoevolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sub_population_size = 10 * dim  # Sub-population size for diversity
        self.num_sub_populations = 2  # Number of sub-populations
        self.F = 0.7  # Mutation factor
        self.CR = 0.8  # Crossover probability
        self.populations = [None] * self.num_sub_populations
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.elitist_archive = []

    def initialize_populations(self, lb, ub):
        for i in range(self.num_sub_populations):
            self.populations[i] = np.random.uniform(lb, ub, (self.sub_population_size, self.dim))

    def evaluate_population(self, func, population):
        scores = np.apply_along_axis(func, 1, population)
        self.evaluations += len(scores)
        return scores

    def cooperative_step(self, func, scores, population, lb, ub):
        for i in range(self.sub_population_size):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.sub_population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, lb, ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            f_trial = func(trial)
            self.evaluations += 1
            if f_trial < scores[i]:
                scores[i] = f_trial
                population[i] = trial
                if f_trial < self.best_score:
                    self.best_score = f_trial
                    self.best_solution = trial
                    self.elitist_archive.append(trial)
                    if len(self.elitist_archive) > 10:
                        self.elitist_archive.pop(0)

    def periodicity_bias(self, population, lb, ub):
        period = np.mean(self.elitist_archive, axis=0) if self.elitist_archive else np.ones(self.dim) * ((lb + ub) / 2)
        biased_population = population + np.sin(2 * np.pi * (population - period))
        return np.clip(biased_population, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_populations(lb, ub)
        while self.evaluations < self.budget:
            for i in range(self.num_sub_populations):
                scores = self.evaluate_population(func, self.populations[i])
                self.cooperative_step(func, scores, self.populations[i], lb, ub)
                self.populations[i] = self.periodicity_bias(self.populations[i], lb, ub)
        return self.best_solution