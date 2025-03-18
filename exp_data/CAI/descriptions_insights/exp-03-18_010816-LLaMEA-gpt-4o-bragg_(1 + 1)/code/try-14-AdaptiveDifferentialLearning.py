import numpy as np

class AdaptiveDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(4 + int(3 * np.log(self.dim)), 50)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1  # New parameter for elitism
        
    def __call__(self, func):
        bounds = func.bounds
        lower_bound, upper_bound = bounds.lb, bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            # Dynamic population resizing
            self.population_size = min(self.population_size + 1, 100)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Differential Mutation
                mutant = np.clip(a + self.mutation_factor * (b - c), lower_bound, upper_bound)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                # Selection with stochastic ranking
                trial_fitness = func(trial)
                rank = np.random.rand()
                if trial_fitness < fitness[i] or rank < 0.1:
                    population[i], fitness[i] = trial, trial_fitness
                    self.mutation_factor = 0.5 + 0.5 * np.random.rand() # Adjusting mutation factor

            # Adaptive learning to adjust mutation factor and crossover rate
            self.crossover_rate = 0.6 + 0.4 * np.random.rand()
            # Elitism: retain the top solutions
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            population[:elite_count] = population[elite_indices]
            fitness[:elite_count] = fitness[elite_indices]
        
        best_index = np.argmin(fitness)
        return population[best_index]