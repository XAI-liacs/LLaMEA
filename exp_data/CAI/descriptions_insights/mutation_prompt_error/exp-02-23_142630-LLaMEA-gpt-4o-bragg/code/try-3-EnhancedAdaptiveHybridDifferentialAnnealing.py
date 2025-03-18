import numpy as np

class EnhancedAdaptiveHybridDifferentialAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.temperature = 1.0
        self.alpha = 0.95  # Cooling schedule
        self.mutation_decay = 0.99  # Decay for mutation factor
        self.history = []

    def mutate(self, individuals):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = individuals[idxs]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def acceptance_probability(self, current_value, candidate_value):
        if candidate_value < current_value:
            return 1.0
        else:
            return np.exp((current_value - candidate_value) / self.temperature)

    def dynamic_population_size(self, generation):
        # Increase population size dynamically every few generations
        if generation % 10 == 0 and self.population_size < self.initial_population_size * 2:
            self.population_size += 1

    def adjust_temperature(self, fitness):
        # Adjust temperature based on neighborhood fitness variance
        variance = np.var(fitness)
        self.temperature = max(0.1, self.temperature * np.exp(-0.05 * variance))

    def __call__(self, func):
        self.bounds = func.bounds
        individuals = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in individuals])
        self.history.extend(fitness)

        for gen in range(self.budget // self.population_size):
            self.dynamic_population_size(gen)
            for i in range(self.population_size):
                mutant = self.mutate(individuals)
                trial = self.crossover(individuals[i], mutant)
                trial_fitness = func(trial)
                self.history.append(trial_fitness)

                if self.acceptance_probability(fitness[i], trial_fitness) > np.random.rand():
                    individuals[i] = trial
                    fitness[i] = trial_fitness

            self.adjust_temperature(fitness)
            # Adaptive mutation factor adjustment
            best_fitness = np.min(fitness)
            mean_fitness = np.mean(fitness)
            if best_fitness < mean_fitness:
                self.mutation_factor *= self.mutation_decay

            # Cooling down the temperature
            self.temperature *= self.alpha

        best_idx = np.argmin(fitness)
        return individuals[best_idx], fitness[best_idx], self.history