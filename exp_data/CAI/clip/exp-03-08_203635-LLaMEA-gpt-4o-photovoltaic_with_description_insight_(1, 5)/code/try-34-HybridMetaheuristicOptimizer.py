import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, iters, pop_size=50, F=0.8, CR=0.9):
        pop_size = max(8, pop_size // 2)  # Adjusted initial pop_size
        population = np.random.rand(pop_size, self.dim)
        population *= (bounds.ub - bounds.lb)
        population += bounds.lb
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        for _ in range(iters):
            if self.evaluations >= self.budget:
                break
            F = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor
            CR = np.random.uniform(0.6, 0.9)  # Adaptive crossover rate
            ranked_indices = np.argsort(fitness)  # Rank population
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                if i < pop_size // 2:
                    a, b, c = population[ranked_indices[np.random.choice(pop_size // 2, 3, replace=False)]]
                else:
                    a, b, c = population[np.random.choice(population.shape[0], 3, replace=False)]
                
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
            
            if pop_size < 100:
                new_individual = np.random.rand(1, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
                population = np.vstack([population, new_individual])
                fitness = np.append(fitness, func(new_individual))
        best_idx = np.argmin(fitness)
        return population[best_idx], np.min(fitness)

    def local_refinement(self, func, bounds, x0):
        result = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        iterations = self.budget // 10
        global_solution, global_cost = self.differential_evolution(func, bounds, iters=iterations)
        local_solution, local_cost = self.local_refinement(func, bounds, global_solution)
        
        if local_cost < global_cost:
            return local_solution
        else:
            return global_solution