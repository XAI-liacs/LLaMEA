import numpy as np
from scipy.optimize import minimize, basinhopping

class EnhancedHybridDEBH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.evaluations = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
    
    def enforce_adaptive_periodicity(self, individual):
        if np.random.rand() < 0.5:
            half_point = self.dim // 2
            for i in range(half_point):
                avg_value = (individual[i] + individual[i + half_point]) / 2
                individual[i] = avg_value
                individual[i + half_point] = avg_value
        return individual
    
    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')

        while self.evaluations < self.budget:
            new_population = []
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial = self.enforce_adaptive_periodicity(trial)
                trial_value = func(trial)
                self.evaluations += 1
                if trial_value > func(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
                if trial_value > best_value:
                    best_value = trial_value
                    best_solution = trial
                if self.evaluations >= self.budget:
                    break
            population = np.array(new_population)
            if self.evaluations >= self.budget:
                break
        return best_solution
    
    def stochastic_local_search(self, func, candidate, bounds):
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": list(zip(bounds.lb, bounds.ub))}
        result = basinhopping(func, candidate, minimizer_kwargs=minimizer_kwargs, niter=100, disp=False)
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        best_candidate = self.differential_evolution(func, bounds)
        refined_solution, refined_value = self.stochastic_local_search(func, best_candidate, bounds)
        return refined_solution