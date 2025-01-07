import numpy as np

class QIADE:
    def __init__(self, budget, dim, population_size=50, crossover_rate=0.9, beta_min=0.2, beta_max=0.8, gamma=0.99):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma = gamma
        self.evaluations = 0
        self.adaptive_mutation = lambda t: self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * t)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_position = None
        best_value = float('inf')
        mutation_factor = self.beta_max

        while self.evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                x1, x2, x3 = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant_vector = x1 + mutation_factor * (x2 - x3)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_value = func(trial_vector)
                self.evaluations += 1

                if trial_value < func(population[i]):
                    new_population.append(trial_vector)
                    if trial_value < best_value:
                        best_value = trial_value
                        best_position = trial_vector
                else:
                    new_population.append(population[i])

            population = np.array(new_population)
            mutation_factor = self.adaptive_mutation(self.evaluations / self.budget)

        return best_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))