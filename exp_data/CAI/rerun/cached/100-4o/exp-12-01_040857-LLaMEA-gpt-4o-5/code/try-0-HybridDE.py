import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, 10 * dim)  # Adjust population size based on dimension
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                target = self.population[i]
                a, b, c = self.select_random_indices(i)
                mutant = self.mutate(a, b, c)
                trial = self.crossover(target, mutant)
                trial = np.clip(trial, self.bounds[0], self.bounds[1])
                self.evaluate_and_select(func, i, trial)
            self.population = new_population if self.evaluations < self.budget else self.population
        best_index = np.argmin(self.fitness)
        return self.population[best_index]

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

    def select_random_indices(self, current_index):
        indices = list(range(self.pop_size))
        indices.remove(current_index)
        selected = np.random.choice(indices, 3, replace=False)
        return self.population[selected[0]], self.population[selected[1]], self.population[selected[2]]

    def mutate(self, a, b, c):
        return a + self.f * (b - c)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def evaluate_and_select(self, func, target_index, trial):
        trial_fitness = func(trial)
        self.evaluations += 1
        if trial_fitness < self.fitness[target_index]:
            self.fitness[target_index] = trial_fitness
            self.population[target_index] = trial