import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.min_population_size = 10
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.position = None
        self.best_solution = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i, score in enumerate(scores):
            if score < self.best_score:
                self.best_score = score
                self.best_solution = self.position[i]
        return scores

    def mutate(self, idx, scores):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.position[a] + self.F * (self.position[b] - self.position[c])

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == np.random.randint(self.dim):
                trial[j] = mutant[j]
        return trial

    def adaptive_resize(self, iteration, max_iterations):
        if iteration % (max_iterations // 4) == 0 and iteration > 0:
            self.population_size = max(self.min_population_size, self.population_size // 2)
            self.position = self.position[:self.population_size]

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0

        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            
            new_population = np.copy(self.position)
            for i in range(self.population_size):
                mutant = self.mutate(i, scores)
                trial = self.crossover(self.position[i], mutant)
                trial_score = func(trial)
                func_calls += 1

                if trial_score < scores[i]:
                    new_population[i] = trial

            self.position = new_population
            self.adaptive_resize(iteration, max_iterations)
            iteration += 1

        return self.best_solution, self.best_score