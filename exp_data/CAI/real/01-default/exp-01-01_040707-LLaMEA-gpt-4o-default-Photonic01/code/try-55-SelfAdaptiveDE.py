import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.position = None
        self.bounds = None
        self.fitness = None

    def initialize(self, bounds):
        self.bounds = np.array([bounds.lb, bounds.ub])
        self.position = np.random.rand(self.population_size, self.dim)
        self.position = self.bounds[0] + self.position * (self.bounds[1] - self.bounds[0])
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.fitness[i]:
                self.fitness[i] = scores[i]
        return scores

    def mutate(self, target_idx):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.position[np.random.choice(candidates, 3, replace=False)]
        mutant_vector = a + self.mutation_factor * (b - c)
        return np.clip(mutant_vector, self.bounds[0], self.bounds[1])

    def crossover(self, target, mutant):
        crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return crossover_vector

    def adapt_parameters(self, iteration, max_iterations):
        self.mutation_factor = 0.5 + 0.3 * np.sin(np.pi * iteration / max_iterations)
        self.crossover_rate = 0.9 - 0.4 * np.sin(np.pi * iteration / max_iterations)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        best_score = float('inf')
        best_solution = None

        for iteration in range(max_iterations):
            self.adapt_parameters(iteration, max_iterations)
            scores = self.evaluate(func)
            func_calls += self.population_size

            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.position[i], mutant)
                trial_score = func(trial)
                func_calls += 1

                if trial_score < self.fitness[i]:
                    self.position[i] = trial
                    self.fitness[i] = trial_score

                if trial_score < best_score:
                    best_score = trial_score
                    best_solution = trial

                if func_calls >= self.budget:
                    break

            if func_calls >= self.budget:
                break

        return best_solution, best_score