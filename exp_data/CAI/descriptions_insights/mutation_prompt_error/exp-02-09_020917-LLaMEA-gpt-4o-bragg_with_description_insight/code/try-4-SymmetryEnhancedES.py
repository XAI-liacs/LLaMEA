import numpy as np
from scipy.optimize import minimize

class SymmetryEnhancedES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None
        self.population_size = 8 * dim
        self.mutation_strength = 0.1
        self.best_solution = None
        self.best_score = np.inf

    def initialize_population(self, lb, ub):
        middle = (ub + lb) / 2
        deviation = (ub - lb) / 4
        population = middle + deviation * np.random.randn(self.population_size, self.dim)
        return np.clip(population, lb, ub)

    def periodic_and_symmetric_constraint(self, position):
        period = (self.ub - self.lb) / self.dim
        period_position = self.lb + (np.round((position - self.lb) / period) * period)
        sym_position = (period_position + np.flip(period_position)) / 2
        return np.clip(sym_position, self.lb, self.ub)

    def apply_mutation(self, population):
        noise = self.mutation_strength * np.random.randn(*population.shape)
        mutated_population = population + noise
        constrained_population = np.array([self.periodic_and_symmetric_constraint(ind) for ind in mutated_population])
        return constrained_population

    def evolutionary_strategy(self, func):
        population = self.initialize_population(self.lb, self.ub)
        for _ in range(self.budget - self.population_size):
            scores = np.array([func(ind) for ind in population])
            best_index = np.argmin(scores)
            if scores[best_index] < self.best_score:
                self.best_solution = population[best_index]
                self.best_score = scores[best_index]
            selected_indices = scores.argsort()[:self.population_size // 2]
            selected_population = population[selected_indices]
            offspring_population = self.apply_mutation(selected_population)
            population = np.vstack([selected_population, offspring_population])

    def local_refinement(self, func):
        result = minimize(func, self.best_solution, method='BFGS', bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            self.best_solution = result.x
            self.best_score = func(result.x)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.evolutionary_strategy(func)
        self.local_refinement(func)
        return self.best_solution