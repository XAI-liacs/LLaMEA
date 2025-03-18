import numpy as np
from scipy.optimize import minimize

class InertiaMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.base_mutation_factor = 0.5
        self.crossover_rate = 0.85
        self.periodicity_factor = 0.1
        self.local_search_prob = 0.25
        self.best_solution = None
        self.inertia_weight = 0.9
        self.velocity = np.zeros((self.pop_size, self.dim))

    def periodicity_cost(self, solution):
        periodic_error = np.var(solution[::2] - solution[1::2]) + np.var(solution - np.roll(solution, 3))
        return periodic_error

    def initialize_population(self, lb, ub):
        return lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)

    def adapt_inertia_weight(self, generation):
        return 0.5 + 0.4 * np.exp(-generation / 10)

    def mutate(self, population, idx, generation):
        indices = np.random.choice(range(self.pop_size), 3, replace=False)
        while idx in indices:
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
        a, b, c = population[indices]
        mutation_factor = self.base_mutation_factor
        mutant = a + mutation_factor * (b - c)
        return np.clip(mutant, 0, 1)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func):
        target_cost = func(target) + self.periodicity_factor * self.periodicity_cost(target)
        trial_cost = func(trial) + self.periodicity_factor * self.periodicity_cost(trial)
        return trial if trial_cost < target_cost else target

    def local_refinement(self, solution, func, lb, ub):
        res = minimize(func, solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return res.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            new_population = []
            inertia_weight = self.adapt_inertia_weight(generation)
            for idx in range(self.pop_size):
                target = population[idx]
                mutant = self.mutate(population, idx, generation)
                trial = self.crossover(target, mutant)
                trial = lb + trial * (ub - lb)
                selected = self.select(target, trial, func)
                self.velocity[idx] = inertia_weight * self.velocity[idx] + (1 - inertia_weight) * (selected - target)
                selected = selected + self.velocity[idx]
                selected = np.clip(selected, lb, ub)
                if np.random.rand() < self.local_search_prob:
                    selected = self.local_refinement(selected, func, lb, ub)
                new_population.append(selected)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.array(new_population)
            generation += 1

        best_idx = np.argmin([func(ind) + self.periodicity_factor * self.periodicity_cost(ind) for ind in population])
        best_solution = population[best_idx]

        if self.best_solution is None or func(best_solution) < func(self.best_solution):
            self.best_solution = best_solution

        result = minimize(lambda x: func(x) + self.periodicity_factor * self.periodicity_cost(x),
                          self.best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return result.x