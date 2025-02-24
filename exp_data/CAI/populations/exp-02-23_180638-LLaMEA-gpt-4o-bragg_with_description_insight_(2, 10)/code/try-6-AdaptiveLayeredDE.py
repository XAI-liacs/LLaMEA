import numpy as np

class AdaptiveLayeredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = None
        self.lb = None
        self.ub = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.layer_groups = [1] * dim

    def initialize_population(self):
        self.population = self.lb + np.random.rand(self.population_size, self.dim) * (self.ub - self.lb)

    def evaluate_population(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()

    def differential_evolution_step(self, func):
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + 0.8 * (b - c), self.lb, self.ub)

            group_sizes = np.unique(self.layer_groups, return_counts=True)[1]
            sorted_indices = np.argsort(group_sizes)
            trial_vector = self.population[i].copy()

            for group_size in sorted_indices:
                change_indices = np.where(self.layer_groups == group_size)[0]
                if np.random.rand() < 0.9:
                    trial_vector[change_indices] = mutant_vector[change_indices]

            trial_fitness = func(trial_vector)
            if trial_fitness > self.fitness[i]:
                self.population[i] = trial_vector
                self.fitness[i] = trial_fitness
                if trial_fitness > self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector.copy()

    def adapt_layer_groups(self):
        group_sizes = np.unique(self.layer_groups, return_counts=True)[1]
        most_common_size = np.argmax(group_sizes)
        for i in range(len(self.layer_groups)):
            if np.random.rand() < 0.1:
                self.layer_groups[i] = most_common_size

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population()
        self.evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.differential_evolution_step(func)
            self.adapt_layer_groups()
            self.evaluate_population(func)
            evaluations += self.population_size

        return self.best_solution