import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def differential_evolution(self, bounds, pop_size=20, F=0.8, CR=0.9):
        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        best_solution = None
        best_value = float('inf')
        
        def mutate(target_idx):
            idxs = [idx for idx in range(pop_size) if idx != target_idx]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            return np.clip(a + F * (b - c), bounds.lb, bounds.ub)

        def crossover(target, mutant):
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, target)
            return trial

        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                mutant = mutate(i)
                trial = crossover(population[i], mutant)
                trial_value = self.func(trial)

                if trial_value < best_value:
                    best_value = trial_value
                    best_solution = trial

                if trial_value < self.func(population[i]):
                    population[i] = trial

        return best_solution, best_value

    def local_search(self, solution):
        res = minimize(self.func, solution, bounds=[(lb, ub) for lb, ub in zip(self.bounds.lb, self.bounds.ub)], method='L-BFGS-B')
        return res.x

    def __call__(self, func):
        self.func = func
        self.bounds = func.bounds

        best_solution, best_value = self.differential_evolution(self.bounds)
        fine_tuned_solution = self.local_search(best_solution)

        if self.func(fine_tuned_solution) < best_value:
            best_solution = fine_tuned_solution

        return best_solution

# Example usage:
# optimizer = BraggMirrorOptimizer(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_func)