import numpy as np
from scipy.stats import levy

class AdaptiveDifferentialEvolutionLevyChaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.position = None
        self.scores = None
        self.gbest = None
        self.gbest_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.scores[i]:
                self.scores[i] = scores[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def levy_flight(self, step_size=0.01):
        return levy.rvs(size=self.dim) * step_size

    def chaotic_perturbation(self, iteration, max_iterations, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        chaos_factor = np.sin(2 * np.pi * iteration / max_iterations)
        perturbation = (ub - lb) * chaos_factor * np.random.rand(self.dim)
        return perturbation

    def mutate_and_recombine(self, bounds, iteration, max_iterations):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_position = np.copy(self.position)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = self.position[np.random.choice(indices, 3, replace=False)]
            mutant = a + self.F * (b - c) + self.levy_flight()
            mutant = np.clip(mutant, lb, ub)
            trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.position[i])
            trial += self.chaotic_perturbation(iteration, max_iterations, bounds)
            new_position[i] = np.clip(trial, lb, ub)
        return new_position

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.position = self.mutate_and_recombine(func.bounds, iteration, max_iterations)
            iteration += 1

        return self.gbest, self.gbest_score