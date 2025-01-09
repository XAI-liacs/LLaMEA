import numpy as np

class AdaptiveEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(5 * np.log(dim))
        self.mutation_factor = 0.8
        self.recombination_rate = 0.9
        self.population = None
    
    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        
    def differential_mutation(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant
    
    def recombine(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.recombination_rate
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, trial, target, func):
        if func(trial) < func(target):
            return trial
        else:
            return target
    
    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self.differential_mutation(i)
                trial = self.recombine(target, mutant)
                
                target_fitness = func(target)
                evaluations += 1
                if evaluations >= self.budget:
                    break
                
                trial_fitness = func(trial)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                self.population[i] = self.select(trial, target, lambda x: (trial_fitness if np.array_equal(x, trial) else target_fitness))
        
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]