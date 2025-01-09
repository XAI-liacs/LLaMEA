import numpy as np

class DEABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = -5.0, 5.0
        pop_size = 20
        de_iter = 100
        abc_iter = 50
        
        def de(x, f, cr=0.7, f_weight=0.5):
            mutant = x + f_weight * (x[np.random.permutation(len(x))] - x[np.random.permutation(len(x))])
            trial = np.where(np.random.rand(len(x)) < cr, mutant, x)
            return trial if f(trial) <= f(x) else x
        
        def abc_search(x):
            employed_bees = np.random.uniform(lb, ub, size=(pop_size, self.dim))
            fitness = np.array([f(bee) for bee in employed_bees])
            best_bee = employed_bees[np.argmin(fitness)]
            
            for _ in range(abc_iter):
                new_bees = employed_bees + np.random.uniform(-1, 1, size=(pop_size, self.dim)) * (employed_bees - best_bee)
                new_fitness = np.array([f(bee) for bee in new_bees])
                
                improved_idx = new_fitness < fitness
                employed_bees[improved_idx] = new_bees[improved_idx]
                fitness[improved_idx] = new_fitness[improved_idx]
                
                best_bee = employed_bees[np.argmin(fitness)]
            
            return best_bee
        
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        for _ in range(self.budget // pop_size):
            for _ in range(de_iter):
                best_solution = de(best_solution, func)
            
            abc_solution = abc_search(best_solution)
            best_solution = abc_solution if func(abc_solution) < func(best_solution) else best_solution
        
        return best_solution