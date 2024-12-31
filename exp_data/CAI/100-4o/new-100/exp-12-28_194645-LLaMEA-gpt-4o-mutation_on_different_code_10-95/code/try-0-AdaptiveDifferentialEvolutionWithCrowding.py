import numpy as np

class AdaptiveDifferentialEvolutionWithCrowding:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.Inf)
        self.evaluations = 0

    def differential_evolution_step(self, func):
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            a, b, c = np.random.choice(self.population_size, 3, replace=False)
            mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, -5.0, 5.0)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            f_trial = func(trial)
            self.evaluations += 1
            if f_trial < self.fitness[i]:
                new_population[i] = trial
                self.fitness[i] = f_trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            if self.evaluations >= self.budget:
                break
        self.population = new_population

    def crowding_distance(self):
        sorted_indices = np.argsort(self.fitness)
        population = self.population[sorted_indices]
        fitness = self.fitness[sorted_indices]
        distances = np.zeros(self.population_size)
        dim_ranges = np.ptp(population, axis=0)
        for i in range(self.dim):
            sorted_dim_index = np.argsort(population[:, i])
            distances[sorted_dim_index[0]] = distances[sorted_dim_index[-1]] = np.inf
            for j in range(1, self.population_size - 1):
                distances[sorted_dim_index[j]] += (population[sorted_dim_index[j + 1], i] - population[sorted_dim_index[j - 1], i]) / dim_ranges[i]
        return distances.argsort()

    def adaptive_mutation_strategy(self):
        self.mutation_factor = 0.5 + 0.3 * np.random.rand()

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evaluations = self.population_size

        indices = self.crowding_distance()
        self.population = self.population[indices]
        self.fitness = self.fitness[indices]

        while self.evaluations < self.budget:
            self.adaptive_mutation_strategy()
            self.differential_evolution_step(func)

        return self.f_opt, self.x_opt