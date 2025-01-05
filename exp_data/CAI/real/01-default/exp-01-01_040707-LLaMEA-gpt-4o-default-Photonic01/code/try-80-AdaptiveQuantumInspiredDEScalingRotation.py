import numpy as np

class AdaptiveQuantumInspiredDEScalingRotation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.F = 0.5
        self.CR = 0.9
        self.rotation_angle = np.pi / 6
        self.position = None
        self.best = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best = self.position[i]
        return scores

    def quantum_mutation(self, target, a, b, c):
        diff = (b - c) * np.cos(self.rotation_angle) + np.random.rand(self.dim) * np.sin(self.rotation_angle)
        return target + self.F * diff

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def update_scaling_factor(self, iteration, max_iterations):
        return self.F * (1 - iteration / max_iterations)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            new_population = np.copy(self.position)
            scores = self.evaluate(func)
            for i in range(self.population_size):
                a, b, c = self.position[np.random.choice(self.population_size, 3, replace=False)]
                mutant = self.quantum_mutation(self.position[i], a, b, c)
                child = self.crossover(self.position[i], mutant)
                new_population[i] = child if func(child) < func(self.position[i]) else self.position[i]
                func_calls += 1
                if func_calls >= self.budget:
                    break
            self.position = new_population
            iteration += 1
            self.F = self.update_scaling_factor(iteration, max_iterations)

        return self.best, self.best_score