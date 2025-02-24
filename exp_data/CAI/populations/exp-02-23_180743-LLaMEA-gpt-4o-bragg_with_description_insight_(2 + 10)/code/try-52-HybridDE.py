import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.6  # Differential weight
        self.CR_initial = 0.9  # Initial Crossover probability
        self.bounds = None

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, target_idx, population):
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        adaptive_factor = 1 - (self.budget / (self.pop_size * self.dim))  # New line
        mutant = np.clip(a + adaptive_factor * self.F * (b - c), self.bounds.lb, self.bounds.ub)  # Modified line
        return mutant

    def crossover(self, target, mutant):
        adaptive_CR = self.CR_initial * (1 - (self.budget / (self.pop_size * self.dim)))  # New line
        cross_points = np.random.rand(self.dim) < adaptive_CR  # Modified line
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_optimize(self, best_solution, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)), options={'maxiter': 100})  # Modified line
        return result.x if result.success else best_solution

    def apply_periodicity(self, solution):
        period = max(2, self.dim // 5)
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

            if self.budget > 0 and fitness[best_idx] < np.min(fitness) * 1.01:
                best_solution = self.local_optimize(best_solution, func)
                self.budget -= 1

        return best_solution