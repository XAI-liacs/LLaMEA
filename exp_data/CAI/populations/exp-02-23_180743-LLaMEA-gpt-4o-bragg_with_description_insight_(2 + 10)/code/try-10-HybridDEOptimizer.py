import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.evaluations = 0
        
    def quasi_oppositional_initialization(self, lb, ub):
        mid = (ub + lb) / 2
        range_ = ub - lb
        population = np.random.rand(self.population_size, self.dim) * range_ + lb
        quasi_opposite_population = mid + (mid - population)
        population = np.vstack((population, quasi_opposite_population))
        return np.clip(population, lb, ub)
        
    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = self.quasi_oppositional_initialization(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population.shape[0]

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size * 2, 3, replace=False)
                x0, x1, x2 = population[indices]
                self.mutation_factor = 0.5 + 0.5 * np.random.rand() # Adaptive mutation factor
                mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                self.evaluations += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial

            # Elitist selection
            best_solution = population[np.argmin(fitness)]
            
        return best_solution

    def local_optimization(self, solution, func, bounds):
        result = minimize(func, solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        return result.x if result.success else solution

    def __call__(self, func):
        best_solution = self.differential_evolution(func, func.bounds)
        best_solution = self.local_optimization(best_solution, func, func.bounds)
        return best_solution