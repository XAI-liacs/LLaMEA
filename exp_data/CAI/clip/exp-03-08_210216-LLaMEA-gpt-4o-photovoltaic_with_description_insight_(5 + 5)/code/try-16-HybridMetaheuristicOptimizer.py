import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution_step(self, population, func, bounds, F=0.8, CR=0.9):
        new_population = np.copy(population)
        for i in range(len(population)):
            if self.evaluations >= self.budget:
                break
            indices = np.random.choice(len(population), 3, replace=False)
            a, b, c = population[indices]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial)
            self.evaluations += 1
            if trial_fitness < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_refinement(self, candidate, func, bounds):
        if self.evaluations >= self.budget:
            return candidate
        result = minimize(func, candidate, method='Nelder-Mead', bounds=bounds,
                          options={'maxiter': 10, 'disp': False})
        self.evaluations += result.nfev
        return result.x if result.success else candidate

    def progressive_layer_optimization(self, func, bounds, max_layers):
        population_size = 10
        current_layers = 10
        population = np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))

        while self.evaluations < self.budget and current_layers <= max_layers:
            population = self.differential_evolution_step(population, func, bounds)
            best_candidate = min(population, key=lambda ind: func(ind))
            self.evaluations += 1
            best_candidate = self.local_refinement(best_candidate, func, bounds)
            self.dim = min(current_layers * 2 + 2, max_layers * 2)  # Enhance layer increment logic
            current_layers += 1

        return best_candidate

    def __call__(self, func):
        bounds = func.bounds
        max_layers = self.dim // 2
        best_solution = self.progressive_layer_optimization(func, bounds, max_layers)
        return best_solution