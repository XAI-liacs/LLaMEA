import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.current_evals = 0
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.current_evals += self.population_size
        
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                scale_factor = np.std(fitness) / (np.var(fitness) + 1e-10)  # Adjust scaling factor calculation
                mutant = np.clip(a + scale_factor * (b - c), lb, ub)
                
                mutant = self._encourage_periodicity(mutant)
                
                crossover = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])
                
                new_fit = func(crossover)
                self.current_evals += 1
                
                if new_fit < fitness[i]:
                    population[i] = crossover
                    fitness[i] = new_fit
                    
                if self.current_evals >= self.budget:
                    break
            
            best_idx = np.argmin(fitness)
            res = minimize(func, population[best_idx], method='L-BFGS-B', bounds=list(zip(lb, ub)))
            if res.fun < fitness[best_idx]:
                population[best_idx] = res.x
                fitness[best_idx] = res.fun
                self.current_evals += res.nfev//3  # Adjust BFGS evals reduction
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def _encourage_periodicity(self, individual):
        period = np.random.choice([2, 4, 6], p=[0.6, 0.3, 0.1])  # Expanded dynamic period choice
        for i in range(0, self.dim, period):
            if i + 1 < self.dim:
                avg = np.mean(individual[i:i+period])
                individual[i:i+period] = avg
        return individual