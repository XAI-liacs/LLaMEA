import numpy as np

class AdaptiveDifferentialEvolutionOBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.evaluations = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def opposition_based_learning(self, population):
        opp_population = self.lower_bound + self.upper_bound - population
        return opp_population

    def mutation(self, population, best_idx):
        indices = np.arange(self.population_size)
        np.random.shuffle(indices)
        for i in range(self.population_size):
            x = population[i]
            a, b, c = population[indices[i:i+3]]
            dynamic_F = self.F + np.random.rand() * 0.1  # Line changed for dynamic F adjustment
            mutant = a + dynamic_F * (b - c)  # Line changed to use dynamic_F
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
            yield x, mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, trial, target, trial_fitness, target_fitness):
        if trial_fitness < target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        self.evaluations += self.population_size

        opp_population = self.opposition_based_learning(population)
        opp_fitness = self.evaluate_population(opp_population, func)
        self.evaluations += self.population_size

        merge_population = np.vstack((population, opp_population))
        merge_fitness = np.concatenate((fitness, opp_fitness))

        order = np.argsort(merge_fitness)
        population = merge_population[order[:self.population_size]]
        fitness = merge_fitness[order[:self.population_size]]

        best_idx = np.argmin(fitness)

        while self.evaluations < self.budget:
            new_population = []
            new_fitness = []

            for target, mutant in self.mutation(population, best_idx):
                trial = self.crossover(target, mutant)
                trial_fitness = func(trial)
                self.evaluations += 1

                new_individual, new_fitness_value = self.select(trial, target, trial_fitness, fitness[best_idx])
                new_population.append(new_individual)
                new_fitness.append(new_fitness_value)

                if self.evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            best_idx = np.argmin(fitness)

        return population[best_idx], fitness[best_idx]