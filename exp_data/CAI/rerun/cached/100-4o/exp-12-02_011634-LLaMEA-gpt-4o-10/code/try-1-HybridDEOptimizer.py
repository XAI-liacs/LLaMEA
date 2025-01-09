import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0

    def evaluate(self, func, candidate):
        if self.evaluations < self.budget:
            fitness = func(candidate)
            self.evaluations += 1
            return fitness
        else:
            return np.inf

    def adapt_opposition(self, candidate):
        return self.lower_bound + self.upper_bound - candidate

    def __call__(self, func):
        # Initial evaluation of the population
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = self.evaluate(func, trial)

                # Replacement
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Adaptive opposition-based learning
                opposition = self.adapt_opposition(self.population[i])
                opposition_fitness = self.evaluate(func, opposition)
                if opposition_fitness < self.fitness[i]:
                    self.population[i] = opposition
                    self.fitness[i] = opposition_fitness
            
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]