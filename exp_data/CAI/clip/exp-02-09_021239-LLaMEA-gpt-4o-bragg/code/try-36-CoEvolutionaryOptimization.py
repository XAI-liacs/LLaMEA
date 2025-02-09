import numpy as np

class CoEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.subpop_count = 5
        self.subpop_size = self.population_size // self.subpop_count
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.migration_rate = 0.1
        self.elite_fraction = 0.1
        self.subpopulations = None
        self.fitnesses = None
        self.bounds = None

    def initialize_subpopulations(self):
        self.subpopulations = [
            np.random.uniform(self.bounds.lb, self.bounds.ub, (self.subpop_size, self.dim))
            for _ in range(self.subpop_count)
        ]
        self.fitnesses = [np.full(self.subpop_size, np.inf) for _ in range(self.subpop_count)]

    def mutate(self, subpop_idx, target_idx):
        idxs = np.random.choice(np.delete(np.arange(self.subpop_size), target_idx), 3, replace=False)
        a, b, c = self.subpopulations[subpop_idx][idxs]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def migrate(self):
        for i in range(self.subpop_count):
            if np.random.rand() < self.migration_rate:
                target_subpop = (i + 1) % self.subpop_count
                elite_idx = np.argsort(self.fitnesses[i])[:int(self.elite_fraction * self.subpop_size)]
                migrants = self.subpopulations[i][elite_idx]
                self.subpopulations[target_subpop][:migrants.shape[0]] = migrants
                self.fitnesses[target_subpop][:migrants.shape[0]] = [np.inf] * migrants.shape[0]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_subpopulations()
        remaining_budget = self.budget
        
        while remaining_budget > 0:
            for subpop_idx in range(self.subpop_count):
                self.fitnesses[subpop_idx] = np.array([func(ind) for ind in self.subpopulations[subpop_idx]])
                remaining_budget -= self.subpop_size

                success_count = 0
                for i in range(self.subpop_size):
                    mutant = self.mutate(subpop_idx, i)
                    trial = self.crossover(self.subpopulations[subpop_idx][i], mutant)
                    trial_fitness = func(trial)
                    remaining_budget -= 1

                    if trial_fitness < self.fitnesses[subpop_idx][i]:
                        self.subpopulations[subpop_idx][i] = trial
                        self.fitnesses[subpop_idx][i] = trial_fitness
                        success_count += 1

                    if remaining_budget <= 0:
                        break

            self.migrate()

        best_subpop_idx = np.argmin([np.min(fitness) for fitness in self.fitnesses])
        best_individual_idx = np.argmin(self.fitnesses[best_subpop_idx])
        return self.subpopulations[best_subpop_idx][best_individual_idx]