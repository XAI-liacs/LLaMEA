import numpy as np

class SelfAdaptiveDEQuantumInspired:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.position = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.adaptive_rate = 0.1

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i, score in enumerate(scores):
            if score < self.scores[i]:
                self.scores[i] = score
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = self.position[i]
        return scores

    def mutation(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = self.position[np.random.choice(indices, 3, replace=False)]
        mutant_vector = a + self.F * (b - c)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def quantum_adaptive(self, iteration, max_iterations):
        adaptive_factor = self.adaptive_rate * ((max_iterations - iteration) / max_iterations)
        self.F = 0.5 + adaptive_factor * (np.random.rand() - 0.5)
        self.CR = 0.9 + adaptive_factor * (np.random.rand() - 0.5)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            for i in range(self.population_size):
                mutant = self.mutation(i)
                trial = self.crossover(self.position[i], mutant)
                trial_score = func(trial)
                func_calls += 1
                if trial_score < self.scores[i]:
                    self.position[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial
                if func_calls >= self.budget:
                    break
            self.quantum_adaptive(iteration, max_iterations)
            iteration += 1

        return self.best_solution, self.best_score