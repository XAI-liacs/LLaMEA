import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor_bounds = (0.5, 1.0)
        self.crossover_prob_bounds = (0.1, 0.9)
        self.population = None
        self.func_evals = 0

    def __call__(self, func):
        self.population = self.initialize_population()
        self.evaluate_population(func)
        
        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutation(i)
                trial_vector = self.crossover(self.population[i], donor_vector)
                self.selection(func, i, trial_vector)

        best_idx = np.argmin(self.population[:, -1])
        return self.population[best_idx, :-1]

    def initialize_population(self):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        return np.hstack((pop, np.zeros((self.pop_size, 1))))  # Last column for fitness

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.func_evals < self.budget:
                self.population[i, -1] = func(self.population[i, :-1])
                self.func_evals += 1

    def mutation(self, target_idx):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = np.random.uniform(*self.scaling_factor_bounds)
        donor_vector = self.population[a, :-1] + F * (self.population[b, :-1] - self.population[c, :-1])
        donor_vector = np.clip(donor_vector, self.lower_bound, self.upper_bound)
        return donor_vector

    def crossover(self, target_vector, donor_vector):
        crossover_prob = np.random.uniform(*self.crossover_prob_bounds)
        crossover_mask = np.random.rand(self.dim) < crossover_prob
        trial_vector = np.where(crossover_mask, donor_vector, target_vector[:-1])
        return trial_vector

    def selection(self, func, target_idx, trial_vector):
        if self.func_evals < self.budget:
            trial_fitness = func(trial_vector)
            self.func_evals += 1
            if trial_fitness < self.population[target_idx, -1]:
                self.population[target_idx, :-1] = trial_vector
                self.population[target_idx, -1] = trial_fitness