import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_fitness = np.inf
        self.best_solution = None
        self.evaluations = 0

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1
    
    def adapt_parameters(self):
        diversity = np.std(self.population, axis=0).mean()
        self.F = 0.5 + 0.5 * np.tanh(diversity - 1.0)
        self.CR = 0.7 + 0.2 * np.tanh(1.0 - diversity)

    def mutate_and_crossover(self, target_idx):
        indices = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
        crossover = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover, mutant, self.population[target_idx])
        return trial

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            self.adapt_parameters()
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                trial = self.mutate_and_crossover(i)
                f_trial = func(trial)
                self.evaluations += 1
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial

        return self.best_fitness, self.best_solution