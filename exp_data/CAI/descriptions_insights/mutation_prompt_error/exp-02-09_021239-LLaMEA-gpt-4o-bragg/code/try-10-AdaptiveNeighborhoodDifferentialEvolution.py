import numpy as np

class AdaptiveNeighborhoodDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.bounds = None
        self.historical_success_rate = 0.5
        self.neighborhood_expansion_rate = 0.1

    def initialize_population(self):
        self.population = np.random.uniform(
            self.bounds.lb, self.bounds.ub, (self.population_size, self.dim)
        )

    def mutate(self, target_idx):
        idxs = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        a, b, c = self.population[idxs]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def adaptive_neighborhood_mutation(self, target_idx):
        neighbors = self.select_neighbors(target_idx)
        centroid = np.mean(self.population[neighbors], axis=0)
        perturbation = np.random.uniform(-self.neighborhood_expansion_rate, self.neighborhood_expansion_rate, self.dim)
        mutant = centroid + perturbation
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def select_neighbors(self, target_idx):
        neighborhood_size = max(3, int(self.neighborhood_expansion_rate * self.population_size))
        distances = np.linalg.norm(self.population - self.population[target_idx], axis=1)
        nearest_neighbors = np.argsort(distances)[:neighborhood_size]
        return nearest_neighbors

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.fitness = np.array([func(ind) for ind in self.population])
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            success_count = 0
            for i in range(self.population_size):
                if np.random.rand() > 0.5:
                    mutant = self.mutate(i)
                else:
                    mutant = self.adaptive_neighborhood_mutation(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                remaining_budget -= 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    success_count += 1

                if remaining_budget <= 0:
                    break

            self.historical_success_rate = 0.8 * self.historical_success_rate + 0.2 * (success_count / self.population_size)
            self.mutation_factor = 0.3 + 0.7 * self.historical_success_rate
            self.crossover_rate = 0.6 + 0.2 * (1 - self.historical_success_rate)
            self.neighborhood_expansion_rate = 0.05 + 0.45 * self.historical_success_rate

            num_elites = int(0.1 * self.population_size)
            elite_indices = np.argsort(self.fitness)[:num_elites]
            self.population[:num_elites] = self.population[elite_indices]
            self.fitness[:num_elites] = self.fitness[elite_indices]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]