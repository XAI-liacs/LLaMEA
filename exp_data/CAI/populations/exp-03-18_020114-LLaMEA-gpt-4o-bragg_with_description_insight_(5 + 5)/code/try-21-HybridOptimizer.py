import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def quasi_opposite_init(self, lb, ub, population_size):
        initial_population = np.random.uniform(lb, ub, (population_size, self.dim))
        opposite_population = lb + ub - initial_population
        return np.vstack((initial_population, opposite_population))
    
    def differential_evolution(self, func, lb, ub):
        population_size = 10
        F = 0.5
        CR = 0.9
        pop = self.quasi_opposite_init(lb, ub, population_size)
        n_pop = pop.shape[0]
        num_evaluations = 0
        best_solution = None
        best_score = float('inf')
        
        while num_evaluations < self.budget:
            for i in range(n_pop):
                indices = [idx for idx in range(n_pop) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c) + 0.01 * (np.random.rand(self.dim) - 0.5), lb, ub)  # Refined mutation
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                score = func(trial)
                num_evaluations += 1
                
                if score < func(pop[i]):
                    pop[i] = trial
                    if score < best_score:
                        best_score = score
                        best_solution = trial
                
                if num_evaluations >= self.budget:
                    break
        
        return best_solution, best_score
    
    def local_refinement(self, func, solution, lb, ub):
        result = minimize(func, solution, bounds=[(lb, ub)]*self.dim, method='L-BFGS-B')
        return result.x, result.fun
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_score = self.differential_evolution(func, lb, ub)
        refined_solution, refined_score = self.local_refinement(func, best_solution, lb, ub)
        return refined_solution if refined_score < best_score else best_solution