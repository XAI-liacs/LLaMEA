import numpy as np
from scipy.optimize import minimize

class HybridDEPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.mutation_factor = 0.9  # Adjusted mutation factor from 0.8 to 0.9
        self.crossover_rate = 0.9
        self.periodicity_factor = 0.1

    def periodicity_cost(self, solution):
        return np.var(solution[:self.dim//2] - solution[self.dim//2:])

    def initialize_population(self, lb, ub):
        return lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)

    def mutate(self, population, idx):
        indices = np.random.choice(range(self.pop_size), 3, replace=False)
        while idx in indices:
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
        a, b, c = population[indices]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, 0, 1)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func, penalty_factor):
        target_cost = func(target) + penalty_factor * self.periodicity_cost(target)
        trial_cost = func(trial) + penalty_factor * self.periodicity_cost(trial)
        return trial if trial_cost < target_cost else target

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            for idx in range(self.pop_size):
                target = population[idx]
                mutant = self.mutate(population, idx)
                trial = self.crossover(target, mutant)
                trial = lb + trial * (ub - lb)
                selected = self.select(target, trial, func, penalty_factor=self.periodicity_factor)
                new_population.append(selected)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.array(new_population)

        # Local refinement on the best solution found
        best_idx = np.argmin([func(ind) + self.periodicity_factor * self.periodicity_cost(ind) for ind in population])
        best_solution = population[best_idx]

        result = minimize(lambda x: func(x) + self.periodicity_factor * self.periodicity_cost(x),
                          best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
                          
        return result.x