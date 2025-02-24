import numpy as np
from scipy.optimize import minimize

class MemeticAlgorithmWithPeriodicityInjection:
    def __init__(self, budget, dim, pop_size=50, F_base=0.8, CR=0.9, period_factor=4):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR = CR
        self.period_factor = period_factor
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def adaptive_mutation(self, population, i):
        indices = np.random.choice(range(self.pop_size), 3, replace=False)
        a, b, c = population[indices]
        F = self.F_base * (1 - self.evaluations / self.budget)
        mutant = np.clip(a + F * (b - c), 0, 1)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def periodicity_injection(self, vector, period):
        pattern = vector[:period]
        return np.tile(pattern, self.dim // period)[:self.dim]

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        self.evaluations += 2
        return candidate if candidate_fitness < target_fitness else target

    def local_search(self, solution, func, bounds):
        def bounded_func(x):
            return func(np.clip(x, bounds[0], bounds[1]))

        result = minimize(bounded_func, solution, method='L-BFGS-B', bounds=bounds)
        return result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        population = self.initialize_population(lb, ub)

        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                target = population[i]
                mutant = self.adaptive_mutation(population, i)
                offspring = self.crossover(target, mutant)

                # Inject periodicity
                period = max(1, self.dim // self.period_factor)
                offspring = self.periodicity_injection(offspring, period)

                population[i] = self.select(offspring, target, func)

                # Local search on best found solution
                if self.evaluations < self.budget:
                    local_solution = self.local_search(population[i], func, bounds)
                    local_fitness = func(local_solution)
                    self.evaluations += 1

                    if local_fitness < best_fitness:
                        best_solution = local_solution
                        best_fitness = local_fitness

        return best_solution