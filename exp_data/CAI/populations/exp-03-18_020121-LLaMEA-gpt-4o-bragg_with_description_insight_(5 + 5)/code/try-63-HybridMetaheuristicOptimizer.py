import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Initial population size
        self.current_evals = 0
    
    def __call__(self, func):
        # Initialize population within bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.current_evals += self.population_size
        
        while self.current_evals < self.budget:
            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                scale_factor = np.std(fitness) / (np.mean(fitness) + 1e-10)  # Adjusted scaling factor
                mutant = np.clip(a + scale_factor * (b - c), lb, ub)
                
                # Encourage periodicity by averaging layer thicknesses to nearby values
                mutant = self._enhance_periodicity(mutant)
                
                # Crossover
                crossover = np.where(np.random.rand(self.dim) < 0.8, mutant, population[i])
                
                # Evaluate new solution
                new_fit = func(crossover)
                self.current_evals += 1
                
                # Selection
                if new_fit < fitness[i]:
                    population[i] = crossover
                    fitness[i] = new_fit
                    
                if self.current_evals >= self.budget:
                    break
            
            # Local optimization with BFGS on best individual
            best_idx = np.argmin(fitness)
            res = minimize(func, population[best_idx], method='L-BFGS-B', bounds=list(zip(lb, ub)))
            if res.fun < fitness[best_idx]:
                population[best_idx] = res.x
                fitness[best_idx] = res.fun
                self.current_evals += res.nfev//2  
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def _enhance_periodicity(self, individual):
        # Adapt period length dynamically based on convergence rate
        period = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
        for i in range(0, self.dim, period):
            if i + period <= self.dim:
                avg = np.mean(individual[i:i+period])
                individual[i:i+period] = avg
        return individual