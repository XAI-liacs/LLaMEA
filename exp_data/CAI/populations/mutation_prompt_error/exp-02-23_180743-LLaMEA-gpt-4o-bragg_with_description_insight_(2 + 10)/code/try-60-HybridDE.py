import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(5, 10 * dim // 2)  # Changed line
        self.F_min, self.F_max = 0.4, 0.9  # Adaptive mutation factors added
        self.CR = 0.9  # Changed line
        self.bounds = None

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, target_idx, population):
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        F = self.F_min + np.random.rand() * (self.F_max - self.F_min)  # Changed line
        mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)  # Changed line
        return mutant

    def crossover(self, target, mutant):
        self.CR = 0.8 + 0.2 * np.random.rand()  # Changed line
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_optimize(self, best_solution, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)), options={'maxiter': 50})  # Changed line
        return result.x if result.success else best_solution

    def apply_periodicity(self, solution):
        period = max(1, self.dim // 4)
        return np.tile(solution[:period], self.dim // period)

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        while self.budget > 0:
            for i in range(self.pop_size):
                mutant = self.mutate(i, population)
                trial = self.crossover(population[i], mutant)
                trial = self.apply_periodicity(trial)
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < fitness[best_idx]:
                    best_idx = i
                    best_solution = trial

                self.budget -= 1
                if self.budget <= 0:
                    break

            if self.budget > 0:
                best_solution = self.local_optimize(best_solution, func)
                self.budget -= 1

        return best_solution