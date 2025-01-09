import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, candidate):
        fitness = func(candidate)
        self.evaluations += 1
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = candidate
        return fitness

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        return self.population[a] + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        fitness = np.array([self.evaluate(func, ind) for ind in self.population])

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = self.evaluate(func, trial)

                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                if self.evaluations >= self.budget:
                    break

        return self.best_solution