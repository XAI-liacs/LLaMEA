import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = min(10, budget // 5)
        self.population = None
        self.function_evaluations = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def opposition_based_population(self, bounds):
        lb, ub = bounds.lb, ub
        return lb + ub - self.population

    def hybrid_opposition_population(self, bounds):
        lb, ub = bounds.lb, ub
        midpoint = (lb + ub) / 2
        return lb + ub - np.random.uniform(lb, ub, (self.population_size, self.dim))  # Modified line

    def mutate(self, target_idx, bounds):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 - (self.function_evaluations / self.budget)**0.5) * np.exp(-self.function_evaluations / self.budget)  # Added exponential cooling adjustment
        adaptive_weight = (1 - self.function_evaluations / self.budget)  # New adaptive weighting mechanism
        mutant = self.population[a] + adaptive_F * adaptive_weight * (self.population[b] - self.population[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        dynamic_CR = self.CR * (1 - self.function_evaluations / self.budget)  # Dynamic control parameter
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            return trial, f_trial
        else:
            return target, f_target

    def __call__(self, func):
        bounds = func.bounds
        self.population_size = min(10, self.budget // 5) + int((self.budget - self.function_evaluations) // 100)  # Dynamic adjustment
        self.population = self.initialize_population(bounds)
        opposition_population = self.opposition_based_population(bounds)
        hybrid_opposition_population = self.hybrid_opposition_population(bounds)
        self.population = np.vstack((self.population, opposition_population, hybrid_opposition_population))
        self.function_evaluations = 0

        fitness = np.array([func(ind) for ind in self.population])
        self.function_evaluations += len(self.population)

        best_idx = np.argmin(fitness)
        best_individual = self.population[best_idx]
        best_fitness = fitness[best_idx]

        while self.function_evaluations < self.budget:
            for i in range(self.population_size):
                if self.function_evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self.mutate(i, bounds)
                trial = self.crossover(target, mutant)

                target, target_fitness = self.select(target, trial, func)
                self.population[i] = target
                fitness[i] = target_fitness
                self.function_evaluations += 1

                if target_fitness < best_fitness:
                    best_fitness = target_fitness
                    best_individual = target

        return best_individual