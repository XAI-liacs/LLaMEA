import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.F_init = 0.5  # Initial mutation factor
        self.CR_init = 0.9  # Initial crossover probability
        self.population = None
        self.evaluate_count = 0
        self.best_solution = None
        self.best_fitness = np.inf

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        # Initialize self-adaptive parameters for each individual
        self.F = np.full(self.population_size, self.F_init)
        self.CR = np.full(self.population_size, self.CR_init)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.evaluate_count += len(self.population)
        return fitness

    def select_parents(self):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        return self.population[idxs]

    def mutate(self, target_idx, bounds):
        a, b, c = self.select_parents()
        current_F = self.F[target_idx] * ((self.budget - self.evaluate_count) / self.budget)**2  # Non-linear decay for F
        mutant = np.clip(a + (0.5 * current_F) * (b - c), bounds.lb, bounds.ub)
        return mutant

    def crossover(self, target, mutant, target_idx):
        current_CR = self.CR[target_idx] * ((self.budget - self.evaluate_count) / self.budget)  # Dynamic adjustment of CR
        cross_points = np.random.rand(self.dim) < current_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def update_parameters(self, idx, success):
        if success:
            self.F[idx] = np.clip(self.F[idx] + 0.05, 0, 1)
            self.CR[idx] = np.clip(self.CR[idx] + 0.1, 0, 1)
        else:
            self.F[idx] = np.clip(self.F[idx] * 0.95, 0, 1)
            self.CR[idx] = np.clip(self.CR[idx] * 0.9, 0, 1)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)

        fitness = self.evaluate_population(func)

        while self.evaluate_count < self.budget:
            self.population_size = max(5, int(10 * self.dim * (1 - self.evaluate_count / self.budget)))
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i, bounds)
                trial = self.crossover(target, mutant, i)
                trial_fitness = func(trial)
                self.evaluate_count += 1

                success = trial_fitness < fitness[i]
                if success:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = target

                self.update_parameters(i, success)

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            self.population = new_population
        return self.best_solution