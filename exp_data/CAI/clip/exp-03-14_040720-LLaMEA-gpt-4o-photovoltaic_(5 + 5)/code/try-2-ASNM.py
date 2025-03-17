import numpy as np

class ASNM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        pop = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        self.memory.append((best_solution, fitness[best_idx]))
        
        evals = population_size
        
        while evals < self.budget:
            # Select top solutions from memory
            top_solutions = sorted(self.memory, key=lambda x: x[1])[:population_size]
            narrowed_lb, narrowed_ub = self._narrow_bounds(top_solutions, lb, ub)
            
            # Generate new population within narrowed bounds
            new_pop = np.random.uniform(narrowed_lb, narrowed_ub, (population_size, self.dim))
            new_fitness = np.apply_along_axis(func, 1, new_pop)
            new_best_idx = np.argmin(new_fitness)
            
            # Update the global best
            if new_fitness[new_best_idx] < fitness[best_idx]:
                best_solution = new_pop[new_best_idx]
                best_idx = new_best_idx
            
            # Update memory
            for i in range(population_size):
                self.memory.append((new_pop[i], new_fitness[i]))
            
            evals += population_size
        
        return best_solution
    
    def _narrow_bounds(self, top_solutions, lb, ub):
        narrowed_lb = np.min([sol[0] for sol in top_solutions], axis=0)
        narrowed_ub = np.max([sol[0] for sol in top_solutions], axis=0)
        
        # Ensure bounds don't collapse
        narrowed_lb = np.maximum(narrowed_lb, lb)
        narrowed_ub = np.minimum(narrowed_ub, ub)
        
        return narrowed_lb, narrowed_ub