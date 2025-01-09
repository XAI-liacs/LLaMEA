import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size based on dimension
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.lower_bound + np.random.rand(self.pop_size, dim) * (self.upper_bound - self.lower_bound)
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def mutation(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial, trial_fitness, fitness):
        if trial_fitness < fitness[target_idx]:
            self.population[target_idx] = trial
            fitness[target_idx] = trial_fitness

    def __call__(self, func):
        fitness = self.evaluate_population(func)
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                mutant = self.mutation(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                eval_count += 1
                self.select(i, trial, trial_fitness, fitness)

        best_idx = np.argmin(fitness)
        return self.population[best_idx]