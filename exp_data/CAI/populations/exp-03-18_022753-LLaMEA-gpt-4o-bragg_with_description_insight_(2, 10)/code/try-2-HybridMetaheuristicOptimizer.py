import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budget_used = 0
    
    def quasi_oppositional_initialization(self, bounds, pop_size):
        X = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        X_opposite = bounds.lb + bounds.ub - X
        return np.concatenate((X, X_opposite), axis=0)
    
    def adaptive_mutation(self, bounds, F_min=0.4, F_max=0.9):
        return np.random.uniform(F_min, F_max)

    def differential_evolution_step(self, population, bounds, CR=0.9):
        new_population = np.empty_like(population)
        pop_size = len(population)
        
        for i in range(pop_size):
            F = self.adaptive_mutation(bounds)
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, population[i])
            new_population[i] = trial
        
        return new_population
    
    def local_search(self, best, func, bounds):
        def periodicity_enforcing(obj):
            penalty = np.cos(2 * np.pi * np.arange(self.dim) / self.dim)
            return obj + 0.1 * np.sum((best - penalty) ** 2)
        
        result = minimize(lambda x: func(x) + periodicity_enforcing(x), best, method='L-BFGS-B', bounds=np.array([bounds.lb, bounds.ub]).T)
        return result.x if result.success else best

    def __call__(self, func):
        bounds = func.bounds
        pop_size = 10
        population = self.quasi_oppositional_initialization(bounds, pop_size)
        population = population[:pop_size]
        fitness = np.array([func(ind) for ind in population])
        self.budget_used += len(population)
        
        while self.budget_used < self.budget:
            trial_population = self.differential_evolution_step(population, bounds)
            trial_fitness = np.array([func(ind) for ind in trial_population])
            self.budget_used += len(trial_population)
            
            for i in range(pop_size):
                if trial_fitness[i] < fitness[i]:
                    population[i], fitness[i] = trial_population[i], trial_fitness[i]
            
            if self.budget_used + 1 < self.budget:
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
                refined_solution = self.local_search(best_solution, func, bounds)
                refined_fitness = func(refined_solution)
                self.budget_used += 1
                
                if refined_fitness < fitness[best_idx]:
                    population[best_idx], fitness[best_idx] = refined_solution, refined_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]