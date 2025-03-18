import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, initial_pop_size, max_iter, F=0.8, CR=0.9):
        pop_size = initial_pop_size * (1 + np.log1p(self.dim))  # Line 1 change for adaptive population size
        pop = np.random.rand(int(pop_size), self.dim)
        pop = bounds.lb + pop * (bounds.ub - bounds.lb)
        pop_fitness = np.array([func(ind) for ind in pop])
        self.evaluations += int(pop_size)

        for iteration in range(max_iter):
            for i in range(int(pop_size)):
                indices = np.random.choice(int(pop_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                F_dynamic = F * (1 - iteration / max_iter)
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), bounds.lb, bounds.ub)
                crossover = np.random.rand(self.dim) < (CR - iteration / (2 * max_iter))
                trial = np.where(crossover, mutant, pop[i])
                
                # Line 2 change for layer-wise perturbation
                if i % 2 == 0: trial = trial + np.random.normal(0, 0.01, self.dim) 

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
        result = minimize(func, solution, method='Nelder-Mead', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)],
                          options={'maxfev': self.budget - self.evaluations, 'adaptive': True})
        self.evaluations += result.nfev
        return result.x if result.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = 25  # Changed from 30 to 25
        max_iter = 100
        phase_budget = self.budget // 2
        
        best_solution = self.differential_evolution(func, bounds, population_size, max_iter)
        if self.evaluations < self.budget:
            best_solution = self.local_refinement(func, best_solution, bounds)
        
        return best_solution