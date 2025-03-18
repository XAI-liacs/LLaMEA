import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.7  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.elite_size = max(1, dim // 10)  # Number of elite solutions to retain

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Sort population by fitness and select elites
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elites = population[elite_indices]
            
            for i in range(self.population_size):
                # Mutation with elite guidance
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                if i in elite_indices:
                    # Guide elites with other elites
                    x1 = x3 = elites[np.random.randint(0, self.elite_size)]
                
                mutant = np.clip(x1 + self.F * (x2 - x3), lower_bound, upper_bound)
                
                # Adaptive crossover rate
                self.CR = np.random.rand()  # Randomly adapt crossover rate
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if evaluations >= self.budget:
                    break
            
            # Dynamic population resizing to focus search
            if evaluations % (self.budget // 10) == 0:
                self.population_size = max(self.elite_size, int(self.population_size * 0.9))
                population = np.concatenate((population[:self.population_size], elites))
                fitness = np.array([func(ind) for ind in population])

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]