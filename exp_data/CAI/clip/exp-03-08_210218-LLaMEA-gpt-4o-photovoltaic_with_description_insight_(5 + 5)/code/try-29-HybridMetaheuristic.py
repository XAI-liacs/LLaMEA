import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, population_size, max_iter, F=0.8, CR=0.9):
        pop = np.random.rand(population_size, self.dim)
        pop = bounds.lb + pop * (bounds.ub - bounds.lb)
        pop_fitness = np.array([func(ind) for ind in pop])
        self.evaluations += population_size
        
        for iteration in range(max_iter):
            CR = 0.5 + 0.5 * (iteration / max_iter)  # Adaptive crossover probability
            dynamic_F = 0.5 + 0.2 * (1 + np.sin(2 * np.pi * iteration / max_iter))  # Improved dynamic mutation factor
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                layer_wise_F = F * (1 + np.random.normal(0, 0.1, self.dim))  # Layer-wise mutation rate
                mutant = np.clip(x0 + dynamic_F * layer_wise_F * (x1 - x2), bounds.lb, bounds.ub)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])
                
                if self.evaluations < self.budget:
                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < pop_fitness[i]:
                        pop[i] = trial
                        pop_fitness[i] = trial_fitness

                if self.evaluations >= self.budget:
                    break
            if self.evaluations >= self.budget:
                break

        return pop[np.argmin(pop_fitness)]

    def local_refinement(self, func, solution, bounds):
        result = minimize(func, solution, method='BFGS', options={'maxiter': self.budget - self.evaluations})
        self.evaluations += result.nfev
        return result.x if result.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = 10 + self.dim  # Dynamic population size
        max_iter = 100
        phase_budget = self.budget // 2
        
        best_solution = self.differential_evolution(func, bounds, population_size, max_iter)
        if self.evaluations < self.budget:
            best_solution = self.local_refinement(func, best_solution, bounds)
        
        return best_solution