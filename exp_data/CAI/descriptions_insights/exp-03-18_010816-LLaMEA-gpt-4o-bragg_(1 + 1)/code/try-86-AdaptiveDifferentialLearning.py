import numpy as np

class AdaptiveDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(4 + int(3 * np.log(self.dim)), 50)
        self.mutation_factor = 1.0  # More aggressive initial mutation factor
        self.crossover_rate = 0.7
    
    def __call__(self, func):
        bounds = func.bounds
        lower_bound, upper_bound = bounds.lb, bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        diversity_factor = 1.0

        for iteration in range(self.budget - self.population_size):
            if iteration % (self.budget // 10) == 0:
                self.population_size = max(4, int(self.population_size * 0.95))  # Gradual reduction
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                adapt_factor = 0.5 + 0.3 * (1 - iteration / self.budget)  # Adaptive part for mutation
                
                mutant = np.clip(a + diversity_factor * self.mutation_factor * adapt_factor * (b - c), lower_bound, upper_bound)
                
                crossover = np.random.rand(self.dim) < self.crossover_rate * adapt_factor  # Adaptive crossover
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    self.mutation_factor = 0.4 + 0.3 * np.random.rand() + 0.3 * (1 - diversity_factor)  # Adjusted line

            self.crossover_rate = 0.5 + 0.5 * (iteration / self.budget)  # Slightly modified adaptive crossover rate
            diversity_factor = 0.9 + 0.05 * np.sin((2 * np.pi * iteration) / self.budget)
            diversity_factor += np.std(fitness) / 50  # Modified adjustment line

            # Line modified below
            self.mutation_factor += 0.05 * np.sin((2 * np.pi * iteration) / self.budget)  # Slight mutation factor adjustment
        
        best_index = np.argmin(fitness)
        return population[best_index]