import numpy as np

class DE_OBL:
    def __init__(self, budget, dim, population_size=30, cr=0.9, f=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.cr = cr
        self.f = f
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __opposition_based_init(self, x):
        return self.lower_bound + self.upper_bound - x
    
    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        while self.budget > 0:
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        mutant[j] = self.__opposition_based_init(mutant[j])
                
                fitness_new = func(mutant)
                if fitness_new < fitness[i]:
                    new_population.append(mutant)
                    fitness[i] = fitness_new
                else:
                    new_population.append(population[i])
                
                self.budget -= 1
                
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]