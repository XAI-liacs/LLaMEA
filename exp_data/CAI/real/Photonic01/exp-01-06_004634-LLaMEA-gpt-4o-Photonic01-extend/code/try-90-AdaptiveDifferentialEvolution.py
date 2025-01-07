import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Increased by 2 * dim for more diversity
        self.mutation_factor = 0.85  # Increased mutation factor
        self.cross_prob = 0.8
        self.population = None
        self.bounds = None
        self.evaluation_count = 0
        self.archive = []  # Introduce archive for historical solutions

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
            self.mutation_factor = min(1.0, self.mutation_factor + 0.04)  # Slightly decreased increment
            if trial not in self.archive:
                self.archive.append(trial)  # Store successful trials
            return trial
        else:
            self.mutation_factor = max(0.4, self.mutation_factor - 0.04)  # Slightly decreased decrement
            return target

    def adapt_parameters(self):
        progress = self.evaluation_count / self.budget
        self.cross_prob = 0.6 + 0.3 * np.sin(np.pi * progress)  # Dynamic crossover based on progress
        self.population_size = max(5, int(8 * self.dim * (1 - progress)))

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
            if self.archive and len(self.archive) > self.population_size // 2:  # Use archive for diversity
                for j in range(len(self.archive) // 4):
                    self.population[np.random.randint(self.population_size)] = self.archive.pop(np.random.randint(len(self.archive)))

        return best_solution