import numpy as np

class EnhancedAdaptiveDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(4 + int(3 * np.log(self.dim)), 50)
        self.mutation_factor = 0.8  # Initial mutation factor adjusted for chaos
        self.crossover_rate = 0.6  # Adjusted initial crossover rate
    
    def __call__(self, func):
        bounds = func.bounds
        lower_bound, upper_bound = bounds.lb, bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        chaos_factor = self._logistic_map(0.7, self.budget)

        for iteration in range(self.budget - self.population_size):
            if iteration % (self.budget // 10) == 0:
                self.population_size = max(4, int(self.population_size * 0.95))  # Gradual reduction
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                adapt_factor = chaos_factor[iteration]  # Chaotic mutation factor
                
                mutant = np.clip(a + self.mutation_factor * adapt_factor * (b - c), lower_bound, upper_bound)
                
                crossover = np.random.rand(self.dim) < np.random.uniform(0.3, 0.8)  # Stochastic adaptive crossover
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    self.mutation_factor = 0.5 + 0.5 * np.random.rand()

            self.crossover_rate = 0.3 + 0.7 * np.sin(iteration / self.budget * np.pi)
            
        best_index = np.argmin(fitness)
        return population[best_index]
    
    def _logistic_map(self, r, size):
        x = np.zeros(size)
        x[0] = 0.5
        for i in range(1, size):
            x[i] = r * x[i-1] * (1 - x[i-1])
        return x