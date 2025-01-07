import numpy as np

class DynamicMemeticDE_AOBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        self._initialize_population(lower_bound, upper_bound)

        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                mutant = self._mutate(i, lower_bound, upper_bound)
                trial = self._crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                self.fitness_evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()

                self._local_search(func, i, lower_bound, upper_bound)

            self._opposition_based_learning(lower_bound, upper_bound, func)

        return self.best_solution

    def _initialize_population(self, lb, ub):
        scale = ub - lb
        self.population = lb + np.random.rand(self.population_size, self.dim) * scale
        for i in range(self.population_size):
            self.fitness[i] = np.inf

    def _mutate(self, index, lb, ub):
        a, b, c = self._select_three_random_indices_excluding(index)
        F = 0.5 + 0.3 * np.random.rand()
        mutant = np.clip(self.population[a] + F * (self.population[b] - self.population[c]), lb, ub)
        return mutant

    def _select_three_random_indices_excluding(self, index):
        indices = list(range(self.population_size))
        indices.remove(index)
        return np.random.choice(indices, 3, replace=False)

    def _crossover(self, target, mutant):
        crossover_rate = 0.9 - 0.5 * (self.fitness_evaluations / self.budget)
        crossover_indices = np.random.rand(self.dim) < crossover_rate
        trial = np.where(crossover_indices, mutant, target)
        return trial

    def _local_search(self, func, index, lb, ub):
        if np.random.rand() < 0.1:
            neighbors = lb + np.random.rand(3, self.dim) * (ub - lb)
            best_neighbor = self.population[index]
            best_fitness = self.fitness[index]
            for neighbor in neighbors:
                neighbor_fitness = func(neighbor)
                self.fitness_evaluations += 1
                if neighbor_fitness < best_fitness:
                    best_neighbor = neighbor
                    best_fitness = neighbor_fitness
            if best_fitness < self.fitness[index]:
                self.population[index] = best_neighbor
                self.fitness[index] = best_fitness

    def _opposition_based_learning(self, lb, ub, func):
        if np.random.rand() < 0.2:
            opposition_population = lb + ub - self.population
            for i, opposition in enumerate(opposition_population):
                if self.fitness_evaluations >= self.budget:
                    break
                opposition_fitness = func(opposition)
                self.fitness_evaluations += 1
                if opposition_fitness < self.fitness[i]:
                    self.population[i] = opposition
                    self.fitness[i] = opposition_fitness
