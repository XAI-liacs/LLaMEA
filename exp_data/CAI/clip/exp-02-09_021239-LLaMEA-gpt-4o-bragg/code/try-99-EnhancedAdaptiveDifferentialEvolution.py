import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.min_population_size = 4
        self.mutation_factor = 0.5
        self.crossover_rate = 0.8
        self.population = None
        self.bounds = None
        self.historical_success_rate = 0.5
        self.elite_fraction = 0.1
        self.chaotic_sequence = self.generate_chaotic_sequence(budget)

    def generate_chaotic_sequence(self, length):
        x = 0.4  # Initial value for logistic map
        sequence = []
        for _ in range(length):
            x = 4.0 * x * (1.0 - x)
            sequence.append(x)
        return np.array(sequence)

    def initialize_population(self):
        self.population = np.random.uniform(
            self.bounds.lb, self.bounds.ub, (self.population_size, self.dim)
        )

    def mutate(self, target_idx, budget_idx):
        idxs = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        a, b, c = self.population[idxs]
        chaotic_factor = self.chaotic_sequence[budget_idx]  # Use chaotic sequence
        mutant = a + chaotic_factor * (b - c)
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def multi_phase_mutation(self, target_idx):
        idxs = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 5, replace=False)
        a, b, c, d, e = self.population[idxs]
        mutant1 = a + self.mutation_factor * (b - c)
        mutant2 = d + self.mutation_factor * (e - a)
        mutant3 = b + self.mutation_factor * (c - d)
        elite = self.population[np.argsort(self.fitness)[:1][0]]
        random_perturbation = np.random.uniform(
            -0.2 * self.historical_success_rate, 0.2 * self.historical_success_rate, self.dim)
        return np.clip((mutant1 + mutant2 + mutant3 + elite) / 4 + random_perturbation, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant):
        diversity = np.mean(np.std(self.population, axis=0))
        adaptive_crossover_rate = self.crossover_rate + 0.1 * diversity
        crossover_mask = np.random.rand(self.dim) < adaptive_crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def resize_population(self):
        diversity = np.mean(np.std(self.population, axis=0))
        success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.population_size
        self.population_size = int(max(self.min_population_size, self.initial_population_size * (0.5 + diversity)))
        self.population_size = min(self.population_size, self.initial_population_size)

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        self.fitness = np.array([func(ind) for ind in self.population])
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            success_count = 0
            for i in range(self.population_size):
                if np.random.rand() > 0.6:
                    mutant = self.mutate(i, self.budget - remaining_budget)
                else:
                    mutant = self.multi_phase_mutation(i)
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
            self.mutation_factor = 0.45 + 0.55 * self.historical_success_rate * np.random.uniform(0.8, 1.2)
            self.crossover_rate = 0.5 + 0.3 * (1 - self.historical_success_rate)
            self.resize_population()

            self.elite_fraction = 0.05 + 0.15 * self.historical_success_rate
            num_elites = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(self.fitness)[:num_elites]
            self.population[:num_elites] = self.population[elite_indices]
            self.fitness[:num_elites] = self.fitness[elite_indices]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]