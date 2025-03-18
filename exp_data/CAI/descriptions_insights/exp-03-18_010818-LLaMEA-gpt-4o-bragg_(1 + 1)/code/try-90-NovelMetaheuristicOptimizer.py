import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = min(10, budget // 5)
        self.num_populations = 3  # Added line
        self.populations = []     # Added line
        self.function_evaluations = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, target_idx, bounds):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * np.random.uniform(0.5, 1.5)  # Changed line for more variability
        adaptive_weight = (1 - self.function_evaluations / self.budget)
        mutant = self.populations[0][a] + adaptive_F * adaptive_weight * (self.populations[0][b] - self.populations[0][c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        dynamic_CR = self.CR * (1 - self.function_evaluations / self.budget)
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

    def random_reinitialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, self.dim)

    def __call__(self, func):
        bounds = func.bounds
        self.populations = [self.initialize_population(bounds) for _ in range(self.num_populations)]  # Changed line
        self.function_evaluations = 0

        fitness_lists = [np.array([func(ind) for ind in pop]) for pop in self.populations]  # Added line
        self.function_evaluations += sum(len(fitness) for fitness in fitness_lists)  # Changed line

        best_fitness = np.min([np.min(fitness) for fitness in fitness_lists])  # Changed line
        best_idx = np.argmin([np.min(fitness) for fitness in fitness_lists])  # Added line
        best_pop = self.populations[np.argmin([np.min(fitness) for fitness in fitness_lists])]  # Added line
        
        best_individual = best_pop[best_idx % self.population_size]  # Added line

        while self.function_evaluations < self.budget:
            for pop_idx in range(self.num_populations):  # Changed line
                for i in range(self.population_size):
                    if self.function_evaluations >= self.budget:
                        break

                    target = self.populations[pop_idx][i]  # Changed line
                    mutant = self.mutate(i, bounds)
                    trial = self.crossover(target, mutant)

                    target, target_fitness = self.select(target, trial, func)
                    self.populations[pop_idx][i] = target  # Changed line
                    fitness_lists[pop_idx][i] = target_fitness  # Changed line
                    self.function_evaluations += 1

                    if target_fitness < best_fitness:
                        best_fitness = target_fitness
                        best_individual = target
                    elif np.random.rand() < min(0.05 * np.std(fitness_lists[pop_idx]) / (np.median(fitness_lists[pop_idx]) + 1e-9), 0.2):
                        self.populations[pop_idx][i] = self.random_reinitialize(bounds)  # Changed line

        return best_individual