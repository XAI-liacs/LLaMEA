import numpy as np

class AGB_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)

    def mutate(self, population, best_idx):
        indices = np.arange(self.pop_size)
        np.random.shuffle(indices)
        idxs = indices[:3]
        while best_idx in idxs:
            np.random.shuffle(indices)
            idxs = indices[:3]
        a, b, c = population[idxs]
        mutant = a + self.f * (b - c)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def gradient_based_adjustment(self, trial, lb, ub, func, step_size=0.01):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            step = np.zeros(self.dim)
            step[i] = step_size
            grad[i] = (func(np.clip(trial + step, lb, ub)) - func(np.clip(trial - step, lb, ub))) / (2 * step_size)
        adjusted = trial - step_size * grad
        return np.clip(adjusted, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        evaluations += self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                best_idx = np.argmin(fitness)
                mutant = self.mutate(population, best_idx)
                trial = self.crossover(population[i], mutant)
                trial = self.gradient_based_adjustment(trial, lb, ub, func)
                trial_value = func(trial)
                evaluations += 1
                
                if trial_value < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_value
                
                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value