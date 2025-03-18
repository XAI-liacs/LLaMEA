import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.evaluate_count = 0

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.evaluate_count += len(self.population)
        return fitness

    def select_parents(self):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        return self.population[idxs]

    def mutate(self, target_idx, bounds):
        a, b, c = self.select_parents()
        current_F = 0.3 + 0.2 * np.sin(np.pi * self.evaluate_count / self.budget)
        random_factor = np.random.uniform(0.9, 1.1)  # Adjusted randomization factor range
        scaling_factor = 1 - (self.evaluate_count / self.budget)  # New dynamic scaling factor
        mutant = np.clip((0.6 * a + 0.2 * b + 0.2 * c) + current_F * (b - c) * random_factor * scaling_factor, bounds.lb, bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        current_CR = self.CR * ((self.budget - self.evaluate_count) / self.budget)**1.5
        cross_points = np.random.rand(self.dim) < current_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        best_solution = None
        best_fitness = np.inf

        fitness = self.evaluate_population(func)

        while self.evaluate_count < self.budget:
            self.population_size = max(5, int(10 * self.dim * (1 - self.evaluate_count / self.budget)))
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i, bounds)
                trial = self.crossover(target, mutant)
                trial_fitness = func(trial)
                self.evaluate_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.F *= 1.05  # Enhance F on success
                else:
                    new_population[i] = target
                    self.CR *= 0.95  # Reduce CR on failure

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

            if self.evaluate_count > self.budget / 4:
                if best_solution is not None:
                    new_population[0] = best_solution

            self.population = new_population
        return best_solution