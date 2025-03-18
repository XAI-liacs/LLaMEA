import numpy as np
from scipy.optimize import minimize

class BraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_budget = 0
    
    def initialize_population(self, pop_size, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(pop_size, self.dim)
    
    def differential_evolution_step(self, pop, bounds, F=0.8, CR=0.9, func=None):  # Added func as an argument
        new_pop = np.copy(pop)
        pop_size = pop.shape[0]
        for i in range(pop_size):
            indices = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
            a, b, c = pop[indices]
            F = 0.5 + 0.3 * np.random.rand()  # Adaptive F
            CR = 0.6 + 0.3 * np.random.rand()  # Adaptive CR
            # Dual-strategy mutation: switch between standard and rand-to-best/1
            if np.random.rand() > 0.5:
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            else:
                best_idx = np.argmin([func(ind) for ind in pop])  # Find the best individual
                mutant = np.clip(a + F * (pop[best_idx] - a) + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            new_pop[i] = trial
        return new_pop

    def local_optimization(self, individual, func, bounds):
        res = minimize(func, individual, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                       method='L-BFGS-B', options={'maxfun': min(100, self.budget-self.current_budget)})
        return res.x, res.fun

    def periodicity_encouragement(self, pop, bounds):
        # Encourage periodic solutions by nudging towards periodic configurations
        period = 2  # Assume known optimal period for this problem
        for i in range(pop.shape[0]):
            pop[i] = np.clip(np.tile(pop[i][:period], self.dim // period + 1)[:self.dim], bounds.lb, bounds.ub)
        return pop

    def __call__(self, func):
        pop_size = 50  # Large enough to explore broadly
        bounds = func.bounds
        population = self.initialize_population(pop_size, bounds)
        best_sol = None
        best_val = float('inf')

        while self.current_budget < self.budget:
            population = self.differential_evolution_step(population, bounds, func=func)  # Pass func to the step
            population = self.periodicity_encouragement(population, bounds)

            for i in range(pop_size):
                if self.current_budget >= self.budget:
                    break
                val = func(population[i])
                self.current_budget += 1
                if val < best_val:
                    best_val = val
                    best_sol = population[i]
            
            # Local refinement if budget allows
            if self.current_budget < self.budget:
                for i in range(pop_size):
                    if self.current_budget >= self.budget:
                        break
                    refined_sol, refined_val = self.local_optimization(population[i], func, bounds)
                    self.current_budget += 1
                    if refined_val < best_val:
                        best_val = refined_val
                        best_sol = refined_sol
                        population[i] = refined_sol
        
        return best_sol