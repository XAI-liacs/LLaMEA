import numpy as np
from scipy.optimize import minimize

class BraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_budget = 0

    def initialize_population(self, pop_size, bounds):
        lb, ub = bounds.lb, bounds.ub
        midpoint = (lb + ub) / 2
        return np.clip(midpoint + (ub - lb) * (np.random.rand(pop_size, self.dim) - 0.5), lb, ub)
    
    def adaptive_population_size(self, initial_size):
        return max(5, initial_size - int(self.current_budget / (self.budget / 10)))

    def differential_evolution_step(self, pop, bounds, F=0.7, CR=0.95):
        new_pop = np.copy(pop)
        pop_size = pop.shape[0]
        for i in range(pop_size):
            indices = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
            a, b, c = pop[indices]
            dynamic_F = F * (1 - (self.current_budget / self.budget))
            mutant = np.clip(a + dynamic_F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            new_pop[i] = trial
        return new_pop

    def local_optimization(self, individual, func, bounds):
        res = minimize(func, individual, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                       method='L-BFGS-B', options={'maxfun': min(50, self.budget-self.current_budget)})
        return res.x, res.fun

    def periodicity_encouragement(self, pop, bounds):
        period = 2
        for i in range(pop.shape[0]):
            pop[i] = np.clip(np.tile(pop[i][:period], self.dim // period + 1)[:self.dim], bounds.lb, bounds.ub)
        return pop

    def symmetry_enforcement(self, pop, bounds):
        for i in range(pop.shape[0]):
            half = self.dim // 2
            pop[i][:half] = (pop[i][:half] + pop[i][half:][::-1]) / 2
            pop[i][half:] = pop[i][:half][::-1]
        return np.clip(pop, bounds.lb, bounds.ub)

    def hybrid_mutation(self, pop, bounds):
        for i in range(pop.shape[0]):
            if np.random.rand() < 0.5:
                indices = np.random.choice(np.arange(pop.shape[0]), 2, replace=False)
                a, b = pop[indices]
                mutant = np.clip(a + 0.5 * (b - a), bounds.lb, bounds.ub)
                pop[i] = mutant
        return pop

    def __call__(self, func):
        initial_pop_size = 50
        bounds = func.bounds
        population = self.initialize_population(initial_pop_size, bounds)
        best_sol = None
        best_val = float('inf')

        while self.current_budget < self.budget:
            population = self.differential_evolution_step(population, bounds)
            population = self.periodicity_encouragement(population, bounds)
            population = self.symmetry_enforcement(population, bounds)
            population = self.hybrid_mutation(population, bounds)

            for i in range(population.shape[0]):
                if self.current_budget >= self.budget:
                    break
                val = func(population[i])
                self.current_budget += 1
                if val < best_val:
                    best_val = val
                    best_sol = population[i]

            if self.current_budget < self.budget:
                for i in range(population.shape[0]):
                    if self.current_budget >= self.budget:
                        break
                    refined_sol, refined_val = self.local_optimization(population[i], func, bounds)
                    self.current_budget += 1
                    if refined_val < best_val:
                        best_val = refined_val
                        best_sol = refined_sol
                        population[i] = refined_sol
        
        return best_sol