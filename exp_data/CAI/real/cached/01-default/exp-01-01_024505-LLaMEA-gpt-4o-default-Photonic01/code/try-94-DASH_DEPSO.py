import numpy as np

class DASH_DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_individuals = max(15, min(60, budget // 12))
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.best_fitness = np.full(self.num_individuals, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.F = 0.5
        self.CR = 0.9
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.chaos_factor = 0.05
        self.mutation_rate = 0.2

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.num_individuals, self.dim)
        self.velocities = np.random.randn(self.num_individuals, self.dim) * 0.1
        self.best_positions = np.copy(self.population)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        for i, f in enumerate(fitness):
            if f < self.best_fitness[i]:
                self.best_fitness[i] = f
                self.best_positions[i] = self.population[i]
            if f < self.global_best_fitness:
                self.global_best_fitness = f
                self.global_best_position = self.population[i]
        return fitness

    def update_population(self, lb, ub):
        for i in range(self.num_individuals):
            indices = np.random.choice(self.num_individuals, 3, replace=False)
            x1, x2, x3 = self.population[indices]
            mutant = x1 + self.F * (x2 - x3)
            mutant = np.clip(mutant, lb, ub)
            crossover = (np.random.rand(self.dim) < self.CR)
            offspring = np.where(crossover, mutant, self.population[i])
            if np.random.rand() < self.mutation_rate:
                offspring = lb + (ub - lb) * np.random.rand(self.dim)
            new_velocity = (self.w * self.velocities[i] +
                            self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.population[i]) +
                            self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.population[i]))
            self.velocities[i] = new_velocity
            self.population[i] = offspring + new_velocity
            self.population[i] = np.clip(self.population[i], lb, ub)

    def apply_chaos_search(self, lb, ub):
        chaotically_perturbed = lb + (ub - lb) * np.random.rand(self.num_individuals, self.dim) * self.chaos_factor
        self.population = np.clip(self.population + chaotically_perturbed, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.num_individuals

            if evaluations >= self.budget:
                break

            self.update_population(lb, ub)
            if evaluations % (self.budget // 10) == 0:
                self.apply_chaos_search(lb, ub)

        return self.global_best_position, self.global_best_fitness