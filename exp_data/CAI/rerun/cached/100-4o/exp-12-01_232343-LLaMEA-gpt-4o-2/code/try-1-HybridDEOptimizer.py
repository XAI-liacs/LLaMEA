import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.CR = 0.9  # Crossover probability
        self.F = 0.5   # Differential weight
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.best = None
        self.best_score = float('inf')
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.pop_size):
                a, b, c = self.select_three_random_indices(i)
                mutant = self.mutate(a, b, c)
                trial = self.crossover(self.population[i], mutant)
                trial_score = func(trial)
                self.eval_count += 1
                new_population[i] = trial if trial_score < func(self.population[i]) else self.population[i]
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best = trial
                if self.eval_count >= self.budget:
                    break
            self.population = new_population

        return self.best

    def select_three_random_indices(self, current_index):
        indices = list(range(self.pop_size))
        indices.remove(current_index)
        return np.random.choice(indices, 3, replace=False)

    def mutate(self, a, b, c):
        self.F = 0.4 + np.random.rand() * 0.6  # Line changed to introduce variability in F
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, self.bounds[0], self.bounds[1])