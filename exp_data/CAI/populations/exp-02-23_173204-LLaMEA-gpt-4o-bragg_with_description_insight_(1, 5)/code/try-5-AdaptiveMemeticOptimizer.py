import numpy as np
from scipy.optimize import minimize

class AdaptiveMemeticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Population size for GA
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.best_solution = None
        self.best_score = float('inf')

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim - 1)
            child = np.concatenate([parent1[:point], parent2[point:]])
            return child
        return parent1

    def mutate(self, solution, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                solution[i] = np.random.uniform(lb[i], ub[i])
        return solution

    def periodic_enforcement(self, solution, period=2):
        return np.repeat(np.mean(solution.reshape(-1, period), axis=1), period)

    def simulated_annealing(self, func, solution, bounds):
        temp = 1.0
        cooling_rate = 0.99
        current_score = func(solution)
        while temp > 1e-3:
            new_solution = solution + np.random.normal(0, 0.1, self.dim)
            new_solution = np.clip(new_solution, bounds.lb, bounds.ub)
            new_solution = self.periodic_enforcement(new_solution)

            new_score = func(new_solution)
            if new_score < current_score or np.exp((current_score - new_score) / temp) > np.random.rand():
                solution, current_score = new_solution, new_score

            temp *= cooling_rate

        return solution, current_score

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds.lb, bounds.ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []

            for _ in range(self.population_size // 2):
                parents = population[np.random.choice(self.population_size, 2, replace=False)]
                child1 = self.crossover(parents[0], parents[1])
                child2 = self.crossover(parents[1], parents[0])

                child1 = self.mutate(child1, bounds.lb, bounds.ub)
                child2 = self.mutate(child2, bounds.lb, bounds.ub)

                child1 = self.periodic_enforcement(child1)
                child2 = self.periodic_enforcement(child2)

                child1, score1 = self.simulated_annealing(func, child1, bounds)
                child2, score2 = self.simulated_annealing(func, child2, bounds)

                new_population.extend([(child1, score1), (child2, score2)])
                evaluations += 2

            population = [sol for sol, _ in sorted(new_population, key=lambda x: x[1])[:self.population_size]]

            if score1 < self.best_score:
                self.best_score, self.best_solution = score1, child1
            if score2 < self.best_score:
                self.best_score, self.best_solution = score2, child2

        return self.best_solution