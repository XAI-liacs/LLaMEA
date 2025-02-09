import numpy as np

class EnhancedHyperMutationDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.bounds = None
        self.success_rate_memory = []
        self.elite_fraction = 0.1

    def initialize_population(self):
        self.population = np.random.uniform(
            self.bounds.lb, self.bounds.ub, (self.population_size, self.dim)
        )

    def mutate(self, target_idx):
        idxs = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        a, b, c = self.population[idxs]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def adaptive_hypermutation(self, target_idx):
        idxs = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 5, replace=False)
        a, b, c, d, e = self.population[idxs]
        mutant1 = a + self.mutation_factor * (b - c)
        mutant2 = d + self.mutation_factor * (e - a)
        random_factor = np.random.uniform(0.5, 1.5)
        hypermutant = random_factor * (mutant1 + mutant2) / 2
        return np.clip(hypermutant, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def resize_population(self, success_rate):
        if success_rate > 0.3:
            self.population_size = int(min(1.5 * self.population_size, self.initial_population_size))
        else:
            self.population_size = int(max(self.population_size // 1.5, self.dim))

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.fitness = np.array([func(ind) for ind in self.population])
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            success_count = 0
            for i in range(self.population_size):
                if np.random.rand() > 0.6:
                    mutant = self.mutate(i)
                else:
                    mutant = self.adaptive_hypermutation(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                remaining_budget -= 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    success_count += 1

                if remaining_budget <= 0:
                    break

            success_rate = success_count / self.population_size
            self.success_rate_memory.append(success_rate)
            if len(self.success_rate_memory) > 5:
                self.success_rate_memory.pop(0)

            avg_success_rate = np.mean(self.success_rate_memory)
            self.mutation_factor = 0.3 + 0.7 * avg_success_rate
            self.crossover_rate = 0.6 + 0.3 * (1 - avg_success_rate)

            self.resize_population(success_rate)
            self.elite_fraction = 0.05 + 0.1 * avg_success_rate
            num_elites = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(self.fitness)[:num_elites]
            self.population[:num_elites] = self.population[elite_indices]
            self.fitness[:num_elites] = self.fitness[elite_indices]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]