import numpy as np

class HybridDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.f_opt = np.Inf
        self.x_opt = None

    def _differential_evolution(self, func, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        func_values = np.array([func(ind) for ind in population])
        self.evaluate_and_update_best(population, func_values)

        for _ in range(self.budget // self.population_size):
            for j in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = indices[indices != j]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover, mutant_vector, population[j])

                trial_value = func(trial_vector)
                if trial_value < func_values[j]:
                    population[j] = trial_vector
                    func_values[j] = trial_value

                self.evaluate_and_update_best(population, func_values)

    def _local_search(self, func, bounds):
        step_size = 0.1
        current = self.x_opt
        current_value = self.f_opt

        for _ in range(10):
            neighbors = [current + step_size * np.random.uniform(-1, 1, self.dim) for _ in range(10)]
            neighbors = [np.clip(neigh, bounds.lb, bounds.ub) for neigh in neighbors]

            for neighbor in neighbors:
                value = func(neighbor)
                if value < current_value:
                    current = neighbor
                    current_value = value

            if current_value < self.f_opt:
                self.f_opt = current_value
                self.x_opt = current

    def evaluate_and_update_best(self, population, func_values):
        min_idx = np.argmin(func_values)
        if func_values[min_idx] < self.f_opt:
            self.f_opt = func_values[min_idx]
            self.x_opt = population[min_idx]

    def __call__(self, func):
        bounds = func.bounds
        self._differential_evolution(func, bounds)
        self._local_search(func, bounds)
        return self.f_opt, self.x_opt