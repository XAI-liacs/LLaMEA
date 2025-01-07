import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.cross_prob = 0.7
        self.population = None
        self.bounds = None
        self.evaluation_count = 0

    def initialize_population(self, lower_bounds, upper_bounds):
        self.population = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
    
    def mutate(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)
    
    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.cross_prob
        trial = np.where(mask, mutant, target)
        return trial

    def select(self, target_idx, trial, func):
        target = self.population[target_idx]
        trial_fitness = func(trial)
        target_fitness = func(target)
        self.evaluation_count += 2
        if trial_fitness < target_fitness:
            return trial
        else:
            return target

    def adapt_parameters(self):
        # Adapt parameters based on evaluation progress
        progress = self.evaluation_count / self.budget
        self.mutation_factor = 0.6 + progress * 0.4  # Changed line
        self.cross_prob = 0.8 * (1 - progress) + 0.2  # Changed line

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)

        best_solution = None
        best_fitness = float('inf')

        while self.evaluation_count < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                self.population[i] = self.select(i, trial, func)
                
                trial_fitness = func(self.population[i])
                self.evaluation_count += 1

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = self.population[i]
                    
                if self.evaluation_count >= self.budget:
                    break

            self.adapt_parameters()

        return best_solution