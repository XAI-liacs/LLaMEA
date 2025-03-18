import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, population_size, max_iter, F=0.8, CR=0.9):
        dynamic_diversity = 1 + 0.5 * ((self.budget - self.evaluations) / self.budget)  # Dynamic diversity adjustment
        pop = np.random.rand(population_size, self.dim) * dynamic_diversity
        pop = bounds.lb + pop * (bounds.ub - bounds.lb)
        pop_fitness = np.array([func(ind) for ind in pop])
        self.evaluations += population_size

        for iteration in range(max_iter):
            CR = 0.5 + 0.5 * (iteration / max_iter)  # Adaptive crossover probability
            dynamic_F = 0.5 + 0.3 * (np.sin(iteration / max_iter * np.pi))  # Dynamic mutation factor
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = np.clip(x0 + dynamic_F * (x1 - x2), bounds.lb, bounds.ub)
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
        remaining_budget = self.budget - self.evaluations
        refinement_budget = min(remaining_budget, 50)  # Fixed budget or remaining evaluations
        noisy_solution = solution + np.random.normal(0, 0.01, self.dim)  # Added Gaussian noise
        result = minimize(func, noisy_solution, method='Nelder-Mead', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)],
                          options={'maxfev': refinement_budget, 'adaptive': True})
        self.evaluations += result.nfev
        return result.x if result.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = int(10 + self.dim * (self.budget - self.evaluations) / self.budget)  # Adjusted population size
        max_iter = 100
        phase_budget = self.budget // 2
        
        best_solution = self.differential_evolution(func, bounds, population_size, max_iter)
        if self.evaluations < self.budget:
            best_solution = self.local_refinement(func, best_solution, bounds)
        
        return best_solution