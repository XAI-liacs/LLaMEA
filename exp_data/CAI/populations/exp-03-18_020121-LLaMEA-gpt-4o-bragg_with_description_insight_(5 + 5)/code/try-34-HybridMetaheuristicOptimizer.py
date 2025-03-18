import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25  # Adjusted population size
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
                F = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                mutant = self._encourage_periodicity(mutant)
                
                crossover_rate = np.random.uniform(0.7, 1.0)  # Adaptive crossover rate
                crossover = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
                
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
                self.current_evals += res.nfev
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def _encourage_periodicity(self, individual):
        period = np.random.choice([2, 4], p=[0.7, 0.3])
        for i in range(0, self.dim, period):
            if i + period <= self.dim:
                avg = np.mean(individual[i:i+period])
                individual[i:i+period] = avg
        return individual