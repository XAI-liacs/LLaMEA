import numpy as np

class HDEALS:
    def __init__(self, budget, dim, pop_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def __call__(self, func):
        func_evals = 0
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in self.population])
        func_evals += self.pop_size
        
        while func_evals < self.budget:
            new_population = np.copy(self.population)
            
            for i in range(self.pop_size):
                # Mutation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])
                
                # Selection
                trial_fitness = func(trial)
                func_evals += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
            
            self.population = new_population
            
            # Adaptive Local Search
            if func_evals < self.budget:
                best_idx = np.argmin(fitness)
                local_search_solution = self.local_search(self.population[best_idx], func)
                local_search_fitness = func(local_search_solution)
                func_evals += 1
                
                if local_search_fitness < fitness[best_idx]:
                    self.population[best_idx] = local_search_solution
                    fitness[best_idx] = local_search_fitness
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx]
                
        return self.best_solution
    
    def local_search(self, solution, func):
        step_size = (self.ub - self.lb) * 0.05
        for _ in range(10):  # Local search iterations
            neighbor = solution + np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(neighbor, self.lb, self.ub)
            if func(neighbor) < func(solution):
                solution = neighbor
        return solution