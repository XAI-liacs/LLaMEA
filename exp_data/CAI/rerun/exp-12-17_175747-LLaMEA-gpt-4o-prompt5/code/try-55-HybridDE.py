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
        self.phase_switch = int(budget * 0.6)  # New parameter for phase-based control

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        prev_best_fitness = np.min(fitness)

        while budget_used < self.budget:
            if budget_used < self.phase_switch:
                self.adjust_diversity(population, fitness)  # New line for diversity control
            if budget_used % (self.budget // 10) == 0:
                self.population_size = max(4, int(self.population_size * 0.95))
            for i in range(self.population_size):
                if np.random.rand() < self.local_search_probability and budget_used > self.phase_switch:  # Condition adjusted
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

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def mutate_and_crossover(self, population, index, fitness):
        indices = [i for i in range(self.population_size) if i != index]
        a, b, c = np.random.choice(indices, size=3, replace=False)
        self.mutation_factor = 0.5 + 0.5 * np.random.rand()  # Adjusted mutation factor range
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        trial = np.copy(population[index])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_probability or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def local_search(self, target, func):
        step_size = 0.05 * (self.upper_bound - self.lower_bound) * np.random.rand()  # Reduced step size for more refined search
        direction = np.random.uniform(-1, 1, self.dim)
        candidate = target + step_size * direction
        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
        return candidate

    def adapt_parameters(self):
        self.crossover_probability = 0.85 + 0.15 * np.random.rand()  # Adjust range of crossover probability

    def adjust_diversity(self, population, fitness):
        std_dev = np.std(population, axis=0)
        excessive_diversity_indices = std_dev > 1.5
        population[:, excessive_diversity_indices] *= 0.9  # Slightly constrain widely spread dimensions