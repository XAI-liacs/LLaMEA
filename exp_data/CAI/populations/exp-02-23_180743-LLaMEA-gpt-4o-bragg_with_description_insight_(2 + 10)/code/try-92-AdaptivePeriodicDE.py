import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(5, 10 * dim // 2)
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.5, 0.9
        self.bounds = None
        self.F_memory = np.random.uniform(self.F_min, self.F_max, self.pop_size)

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, target_idx, population, period_length):
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]

        # Adaptive mutation leveraging periodicity
        if period_length > 1:
            F = self.F_max
        else:
            F = self.F_memory[target_idx]

        mutant = np.clip(a + F * (b - c), self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        CR = self.CR_min + (self.CR_max - self.CR_min) * np.cos(self.budget)
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_optimize(self, best_solution, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x if result.success else best_solution

    def apply_periodicity(self, solution):
        period = max(1, self.dim // 4)
        solution[:period] = np.mean(solution[:period])
        return np.tile(solution[:(self.dim // period) * period].reshape(-1, period).mean(axis=0), self.dim // period)

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        while self.budget > 0:
            for i in range(self.pop_size):
                period_length = self.estimate_periodicity(population[i])
                mutant = self.mutate(i, population, period_length)
                trial = self.crossover(population[i], mutant)
                trial = self.apply_periodicity(trial)
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    self.F_memory[i] = self.F_min + np.random.rand() * (self.F_max - self.F_min)

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

    def estimate_periodicity(self, solution):
        # Simple heuristic for estimating periodicity
        period = 1
        for i in range(1, self.dim // 2):
            if np.allclose(solution[i::i+1], solution[0::i+1]):
                period = i
                break
        return period