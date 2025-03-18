import numpy as np
from scipy.optimize import minimize

class PeriodicDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # heuristic choice for population size
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            if evaluations > self.budget * 0.5:
                self.population_size = max(5, self.population_size // 2)
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                
                self.crossover_prob = 0.5 + 0.5 * (1 - evaluations / self.budget)  # Variable crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                result = minimize(self.enhanced_local_search(func), population[best_idx], bounds=list(zip(lb, ub)))
                if result.success and result.fun < fitness[best_idx]:
                    population[best_idx] = result.x
                    fitness[best_idx] = result.fun
                    evaluations += result.nfev
        
        best_idx = np.argmin(fitness)
        return population[best_idx]

    def enhanced_local_search(self, func):
        def periodic_cost(x):
            cost = func(x)
            periodicity_penalty = np.sum(np.abs(np.diff(x[::2]) - np.diff(x[1::2]))) + np.std(x)
            return cost + periodicity_penalty
        return periodic_cost