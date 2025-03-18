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
        adaptive_F = F

        def mutate(target_idx):
            idxs = [idx for idx in range(pop_size) if idx != target_idx]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            return np.clip(a + adaptive_F * (b - c), bounds.lb, bounds.ub)

        def crossover(target, mutant):
            cross_points = np.random.rand(self.dim) < CR
            periodic_points = np.arange(0, self.dim, 2)
            cross_points[periodic_points] = True
            trial = np.where(cross_points, mutant, target)
            return trial

        def surrogate_sampling():
            surrogate = np.mean(population, axis=0)
            surrogate_value = self.func(surrogate)
            for i in range(pop_size):
                if surrogate_value < self.func(population[i]):
                    population[i] = surrogate

        for gen in range(self.budget // pop_size):
            adaptive_F = 0.5 + (0.5 * np.sin(np.pi * gen / self.budget))  # Adaptive scaling factor
            surrogate_sampling()
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