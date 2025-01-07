import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.local_search_iterations = 5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential evolution mutation
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Local search
                trial = self.local_search(trial, func, lb, ub, self.local_search_iterations)

                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                # Update best position
                if trial_fitness < fitness[best_index]:
                    best_index = i
                    best_position = trial

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def local_search(self, position, func, lb, ub, max_iter):
        best_pos = position
        best_fitness = func(position)
        for _ in range(max_iter):
            candidate = best_pos + 0.01 * (np.random.rand(self.dim) - 0.5) * (ub - lb)
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_pos = candidate
        return best_pos