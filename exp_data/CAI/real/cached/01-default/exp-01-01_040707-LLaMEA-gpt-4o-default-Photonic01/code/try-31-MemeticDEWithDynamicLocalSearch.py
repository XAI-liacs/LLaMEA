import numpy as np

class MemeticDEWithDynamicLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.position = None
        self.scores = None
        self.best_index = None

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        for i in range(self.population_size):
            self.scores[i] = func(self.position[i])
        self.best_index = np.argmin(self.scores)

    def differential_mutation(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        mutant = self.position[r1] + self.F * (self.position[r2] - self.position[r3])
        return np.clip(mutant, func.bounds.lb, func.bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate, func):
        perturbation = (np.random.rand(self.dim) - 0.5) * 0.1 * (func.bounds.ub - func.bounds.lb)
        local_candidate = np.clip(candidate + perturbation, func.bounds.lb, func.bounds.ub)
        return local_candidate if func(local_candidate) < func(candidate) else candidate

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            self.evaluate(func)
            func_calls += self.population_size
            new_population = np.copy(self.position)
            for i in range(self.population_size):
                mutant = self.differential_mutation(i)
                trial = self.crossover(self.position[i], mutant)
                trial = self.local_search(trial, func)
                trial_score = func(trial)
                func_calls += 1
                if trial_score < self.scores[i]:
                    new_population[i] = trial
                    self.scores[i] = trial_score
            self.position = new_population

        best_index = np.argmin(self.scores)
        return self.position[best_index], self.scores[best_index]