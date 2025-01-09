import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.CR = 0.9
        self.F = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.func_evals = 0

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.func_evals += 1
                if self.func_evals >= self.budget:
                    break

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.F * (b - c)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.CR
        trial = np.where(mask, mutant, target)
        return trial

    def select(self, target_idx, trial, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

    def adaptive_control(self):
        diversity = np.mean(np.std(self.population, axis=0))
        self.F = 0.5 + 0.3 * np.tanh(8 * diversity - 2)
        self.CR = 0.5 + 0.4 * np.tanh(6 * (1 - diversity) - 3)

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.func_evals < self.budget:
            self.adaptive_control()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                self.func_evals += 1
                self.select(i, trial, trial_fitness)
                if self.func_evals >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]