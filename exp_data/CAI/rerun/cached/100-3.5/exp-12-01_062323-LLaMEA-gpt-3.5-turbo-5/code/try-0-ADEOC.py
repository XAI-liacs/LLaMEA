import numpy as np

class ADEOC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_pop = 10  # population size
        self.cr = 0.9  # crossover rate
        self.f = 0.8  # differential weight
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        def mutate(x_r1, x_r2, x_r3):
            return x_r1 + self.f * (x_r2 - x_r3)
        
        def clip(x):
            return np.clip(x, self.bounds[0], self.bounds[1])
        
        def orthogonal_crossover(x, v):
            j_rand = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.cr or j == j_rand:
                    x[j] = v[j]
            return x
        
        best_solution = np.random.uniform(*self.bounds, size=self.dim)
        best_fitness = func(best_solution)
        
        population = np.random.uniform(*self.bounds, size=(self.n_pop, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.n_pop):
                indices = np.random.choice(range(self.n_pop), size=3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]
                
                trial_solution = clip(mutate(population[i], x_r1, x_r2))
                trial_solution = orthogonal_crossover(population[i], trial_solution)
                
                trial_fitness = func(trial_solution)
                if trial_fitness < best_fitness:
                    best_solution = np.copy(trial_solution)
                    best_fitness = trial_fitness
            
            population = np.vstack((best_solution, population[1:]))
        
        return best_solution