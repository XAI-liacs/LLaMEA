import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.95
        self.local_search_probability = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        prev_best_fitness = np.min(fitness)

        while budget_used < self.budget:
            self.local_search_probability *= 0.99  # Slight decrease over iterations
            if budget_used % (self.budget // 10) == 0:
                self.population_size = max(4, int(self.population_size * 0.95))  # Adjust population size
            for i in range(self.population_size):
                if np.random.rand() < self.local_search_probability:
                    candidate = self.local_search(population[i], func)
                else:
                    candidate = self.mutate_and_crossover(population, i, fitness)
                candidate_fitness = func(candidate)
                budget_used += 1

                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

                if budget_used >= self.budget:
                    break

            current_best_fitness = np.min(fitness)
            if current_best_fitness < prev_best_fitness:
                prev_best_fitness = current_best_fitness
                self.adapt_parameters()

            # Introduce elitism
            best_index = np.argmin(fitness)
            elite = population[best_index]
            population[np.random.randint(self.population_size)] = elite

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def mutate_and_crossover(self, population, index, fitness):
        indices = [i for i in range(self.population_size) if i != index]
        a, b, c = np.random.choice(indices, size=3, replace=False)
        self.mutation_factor = 0.2 + np.random.rand() * 0.7  # Adjusted mutation factor range
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        trial = np.copy(population[index])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_probability or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def local_search(self, target, func):
        step_size = 0.2 * (self.upper_bound - self.lower_bound) * np.random.rand()  # Enhanced step size
        direction = np.random.uniform(-1, 1, self.dim)
        candidate = target + step_size * direction
        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
        return candidate

    def adapt_parameters(self):
        self.crossover_probability = 0.9 + 0.1 * np.random.rand() * (np.random.rand() > 0.5)
        self.mutation_factor *= 0.97  # Slightly adjusted adaptation rate