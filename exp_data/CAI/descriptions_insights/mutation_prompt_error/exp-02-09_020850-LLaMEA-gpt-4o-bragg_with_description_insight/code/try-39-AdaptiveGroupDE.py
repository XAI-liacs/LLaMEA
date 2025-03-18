import numpy as np
from scipy.optimize import minimize

class AdaptiveGroupDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.base_mutation_factor = 0.5
        self.crossover_rate = 0.85
        self.neutral_zone_factor = 0.1
        self.best_solution = None
        self.group_size = 5

    def initialize_population(self, lb, ub):
        return lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)

    def differential_grouping(self, population):
        random_indices = np.random.permutation(self.pop_size)
        groups = [population[random_indices[i:i + self.group_size]] for i in range(0, self.pop_size, self.group_size)]
        return groups

    def mutate(self, group, idx):
        indices = np.random.choice(len(group), 3, replace=False)
        while idx in indices:
            indices = np.random.choice(len(group), 3, replace=False)
        a, b, c = group[indices]
        mutation_factor = self.base_mutation_factor
        mutant = a + mutation_factor * (b - c) + 0.10 * (np.random.rand(self.dim) - 0.5)
        return np.clip(mutant, 0, 1)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func):
        neutral_zone = np.random.rand(self.dim) < self.neutral_zone_factor
        if np.any(neutral_zone):
            trial[neutral_zone] = target[neutral_zone]
        target_cost = func(target)
        trial_cost = func(trial)
        return trial if trial_cost < target_cost else target

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            groups = self.differential_grouping(population)
            for group in groups:
                for idx in range(len(group)):
                    target = group[idx]
                    mutant = self.mutate(group, idx)
                    trial = self.crossover(target, mutant)
                    trial = lb + trial * (ub - lb)
                    selected = self.select(target, trial, func)
                    new_population.append(selected)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break
                if evaluations >= self.budget:
                    break
            population = np.array(new_population)

        best_idx = np.argmin([func(ind) for ind in population])
        best_solution = population[best_idx]

        if self.best_solution is None or func(best_solution) < func(self.best_solution):
            self.best_solution = best_solution

        result = minimize(func, self.best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return result.x