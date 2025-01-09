import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.elite_fraction = 0.1
        self.neighborhood_size = max(3, int(0.2 * self.population_size))
        self.func_evals = 0

    def __call__(self, func):
        lower, upper = self.bounds
        population = np.random.uniform(lower, upper, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.func_evals += self.population_size

        while self.func_evals < self.budget:
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_set = population[elite_indices]

            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = self.mutate(x1, x2, x3)
                trial = self.crossover(population[i], mutant)

                if np.random.rand() < 0.2:
                    trial = self.local_search(trial, elite_set)

                trial_fitness = func(trial)
                self.func_evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_index = np.argmin(fitness)
        return population[best_index]

    def mutate(self, x1, x2, x3):
        mutant = x1 + self.mutation_factor * (x2 - x3)
        return np.clip(mutant, *self.bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def local_search(self, trial, elite_set):
        for elite in elite_set:
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
            neighbor = trial + perturbation * (elite - trial)
            trial = np.clip(neighbor, *self.bounds)
        return trial