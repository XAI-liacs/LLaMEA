import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def fitness(x):
            return func(x)
        
        def clip(x):
            return np.clip(x, self.lb, self.ub)
        
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_position = population[np.argmin([fitness(ind) for ind in population])]
        best_fitness = fitness(best_position)
        
        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                r1, r2, r3 = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = clip(population[r1] + 0.5 * (population[r2] - population[r3]))
                velocities[i] = 0.9 * velocities[i] + 0.8 * (best_position - population[i]) + 0.5 * (mutant - population[i])
                population[i] = clip(population[i] + velocities[i])
                
                if fitness(population[i]) < fitness(best_position):
                    best_position = population[i]
                    best_fitness = fitness(best_position)
        
        return best_position