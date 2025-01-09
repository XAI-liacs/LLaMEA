import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size based on the dimension
        self.bounds = (-5.0, 5.0)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
    
    def evaluate(self, func):
        for i in range(self.pop_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])

    def select_best(self):
        return np.argmin(self.fitness), np.min(self.fitness)

    def mutation(self, target_idx):
        indices = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, *self.bounds)

    def crossover(self, target, mutant):
        trial = np.array([mutant[j] if np.random.rand() < self.CR else target[j] for j in range(self.dim)])
        return trial

    def cauchy_mutation(self, best):
        scale = 0.2
        return best + scale * np.random.standard_cauchy(self.dim)

    def __call__(self, func):
        evaluations = 0
        self.evaluate(func)
        evaluations += self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                mutant = self.mutation(i)
                target = self.population[i]
                trial = self.crossover(target, mutant)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            best_idx, _ = self.select_best()
            if evaluations < self.budget:
                cauchy_trial = self.cauchy_mutation(self.population[best_idx])
                cauchy_fitness = func(cauchy_trial)
                evaluations += 1

                if cauchy_fitness < self.fitness[best_idx]:
                    self.population[best_idx] = cauchy_trial
                    self.fitness[best_idx] = cauchy_fitness

        best_idx, best_fitness = self.select_best()
        return self.population[best_idx], best_fitness