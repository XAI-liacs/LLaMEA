import numpy as np

class DE_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        f = 0.5
        cr = 0.9
        bounds = (-5.0, 5.0)
        best_solution = np.random.uniform(bounds[0], bounds[1], self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
            for i in range(pop_size):
                target = population[i]
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + f * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, target)
                trial_fitness = func(trial)
                if trial_fitness < best_fitness:
                    best_solution, best_fitness = trial, trial_fitness
        
        return best_solution