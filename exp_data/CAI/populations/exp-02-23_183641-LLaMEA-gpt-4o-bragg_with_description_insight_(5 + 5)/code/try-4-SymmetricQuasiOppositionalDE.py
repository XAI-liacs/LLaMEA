import numpy as np
from scipy.optimize import minimize

class SymmetricQuasiOppositionalDE:
    def __init__(self, budget, dim, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.evaluations = 0

    def quasi_oppositional_initialization(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        pop_opposite = lb + ub - population
        return np.vstack((population, pop_opposite))

    def mutation(self, population):
        indices = np.random.choice(range(self.pop_size), 3, replace=False)
        a, b, c = population[indices]
        mutant = np.clip(a + self.F * (b - c), 0, 1)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

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
        population = self.quasi_oppositional_initialization(lb, ub)
        population = population[:self.pop_size]

        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                target = population[i]
                mutant = self.mutation(population)
                offspring = self.crossover(target, mutant)
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