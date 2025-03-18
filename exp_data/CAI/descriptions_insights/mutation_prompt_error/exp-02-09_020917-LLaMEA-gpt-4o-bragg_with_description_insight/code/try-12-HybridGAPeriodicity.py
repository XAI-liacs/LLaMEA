import numpy as np
from scipy.optimize import minimize

class HybridGAPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.lb = None
        self.ub = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = np.inf
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def initialize_population(self, lb, ub, size):
        self.population = lb + (ub - lb) * np.random.rand(size, self.dim)
        self.pbest_scores = np.full(size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            current_score = func(self.population[i])
            if current_score < self.pbest_scores[i]:
                self.pbest_scores[i] = current_score
            if current_score < self.gbest_score:
                self.gbest = self.population[i].copy()
                self.gbest_score = current_score

    def periodic_crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1, parent2

        crossover_point = np.random.randint(1, self.dim - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return self.periodic_adjustments(child1), self.periodic_adjustments(child2)

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] = self.lb[i] + (self.ub[i] - self.lb[i]) * np.random.rand()
        return self.periodic_adjustments(individual)

    def periodic_adjustments(self, position):
        period = (self.ub - self.lb) / self.dim
        period_position = self.lb + (np.round((position - self.lb) / period) * period)
        return np.clip(period_position, self.lb, self.ub)

    def genetic_algorithm(self, func):
        for _ in range(self.budget - self.population_size):
            new_population = []
            for _ in range(self.population_size // 2):
                parents = np.random.choice(self.population_size, 2, replace=False)
                child1, child2 = self.periodic_crossover(self.population[parents[0]], self.population[parents[1]])
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            self.population = np.array(new_population)
            self.evaluate_population(func)

    def local_refinement(self, func):
        result = minimize(func, self.gbest, method='BFGS', bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            self.gbest = result.x
            self.gbest_score = func(result.x)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub, self.population_size)
        self.evaluate_population(func)
        self.genetic_algorithm(func)
        self.local_refinement(func)
        return self.gbest