import numpy as np

class DE_AMLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.1

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, population):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[indices]
        mutant_vector = a + self.mutation_factor * (b - c)
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _local_search(self, individual):
        perturbation = np.random.normal(0, 0.1, self.dim)
        candidate = individual + perturbation
        return np.clip(candidate, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                mutant_vector = self._mutate(population)
                trial_vector = self._crossover(population[i], mutant_vector)
                if np.random.rand() < self.local_search_prob:
                    trial_vector = self._local_search(trial_vector)

                trial_fitness = func(trial_vector)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                if num_evaluations >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]