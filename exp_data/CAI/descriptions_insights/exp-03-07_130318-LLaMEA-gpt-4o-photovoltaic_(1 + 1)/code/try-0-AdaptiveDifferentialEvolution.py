import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.func_evals = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):  # Ensure at least one parameter is from mutant
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate, bounds):
        step_size = 0.01 * (bounds.ub - bounds.lb)
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        new_candidate = candidate + perturbation
        return np.clip(new_candidate, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds.lb, bounds.ub)
        best_solution = None
        best_fitness = np.inf

        while self.func_evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                trial = self.local_search(trial, bounds)

                trial_fitness = func(trial)
                self.func_evals += 1

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

                if trial_fitness < func(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)

                if self.func_evals >= self.budget:
                    break

            self.population = np.array(new_population)

        return best_solution