import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
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
        current_F = self.F * ((self.budget - self.evaluate_count) / self.budget + 0.5 * (1 - np.min([1.0, np.std(self.evaluate_population(func))])))  # Dynamic adjustment of F
        mutant = np.clip(a + (0.5 * current_F) * (b - c), bounds.lb, bounds.ub)  # Reduced final mutation factor
        return mutant

    def crossover(self, target, mutant):
        current_CR = self.CR * ((self.budget - self.evaluate_count) / self.budget)  # Dynamic adjustment of CR
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
            self.population_size = max(5, int(10 * self.dim * (1 - self.evaluate_count / self.budget)))  # Dynamic population size
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
                else:
                    new_population[i] = target

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

            self.population = new_population
        return best_solution