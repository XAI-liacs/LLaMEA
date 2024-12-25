import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim  # Initial population size
        self.mutation_factor = 0.8
        self.crossover_probability = 0.95
        self.local_search_probability = 0.1

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = population_size

        while budget_used < self.budget:
            for i in range(population_size):
                if np.random.rand() < self.local_search_probability:
                    candidate = self.local_search(population[i], func)
                else:
                    candidate = self.mutate_and_crossover(population, i)
                candidate_fitness = func(candidate)
                budget_used += 1

                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

                if budget_used >= self.budget:
                    break

            population_size = max(4, int(population_size * 0.95))  # Reduce population size dynamically

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def mutate_and_crossover(self, population, index):
        indices = [i for i in range(len(population)) if i != index]
        a, b, c = np.random.choice(indices, size=3, replace=False)
        self.mutation_factor = 0.5 + np.random.rand() * 0.5
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        trial = np.copy(population[index])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_probability or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def local_search(self, target, func):
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * np.random.rand()
        direction = np.random.uniform(-1, 1, self.dim)
        candidate = target + step_size * direction
        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
        return candidate