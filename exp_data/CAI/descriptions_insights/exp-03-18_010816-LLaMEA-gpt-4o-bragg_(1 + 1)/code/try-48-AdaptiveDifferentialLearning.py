import numpy as np

class AdaptiveDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(4 + int(3 * np.log(self.dim)), 50)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        
    def __call__(self, func):
        bounds = func.bounds
        lower_bound, upper_bound = bounds.lb, bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        diversity_factor = 1.0

        learning_rate = np.full(self.population_size, 0.5)  # New: Initialize learning rate per individual
        for iteration in range(self.budget - self.population_size):
            if iteration % (self.budget // 10) == 0:  
                self.population_size = max(4, self.population_size - 1)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
                learning_rate = learning_rate[:self.population_size]  # Adjust learning rate array size

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + diversity_factor * self.mutation_factor * (b - c), lower_bound, upper_bound)
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    learning_rate[i] = 0.5 + 0.5 * np.random.rand()  # New: Update learning rate

            # New: Update mutation factor based on fitness improvement
            self.mutation_factor = np.mean(learning_rate)
            self.crossover_rate = 0.3 + 0.7 * np.sin((np.pi * iteration) / self.budget)
            diversity_factor = 0.9 + 0.1 * np.sin((np.pi * iteration) / self.budget)  
        
        best_index = np.argmin(fitness)
        return population[best_index]