import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.local_search_prob = 0.3
        self.global_perturbation_prob = 0.1
        self.mutation_scale = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.population_size):
            f = func(self.population[i])
            if f < self.fitness[i]:
                self.fitness[i] = f
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = self.population[i]

    def local_search(self, candidate, lb, ub):
        step_size = 0.1 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        new_candidate = np.clip(candidate + perturbation, lb, ub)
        return new_candidate

    def global_perturbation(self, candidate, lb, ub):
        perturbation = np.random.normal(0, self.mutation_scale, self.dim)
        new_candidate = np.clip(candidate + perturbation, lb, ub)
        return new_candidate

    def reproduce(self, parents, lb, ub):
        offspring = np.zeros_like(parents)
        for i in range(self.population_size):
            if np.random.rand() < self.local_search_prob:
                offspring[i] = self.local_search(parents[i], lb, ub)
            else:
                offspring[i] = self.global_perturbation(parents[i], lb, ub)
        return offspring

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            parents = self.population[np.argsort(self.fitness)[:self.population_size // 2]]
            offspring = self.reproduce(parents, lb, ub)
            self.population = np.vstack((parents, offspring))
            self.fitness = np.hstack((self.fitness[np.argsort(self.fitness)[:self.population_size // 2]], np.full(self.population_size // 2, float('inf'))))

        return self.best_solution, self.best_fitness