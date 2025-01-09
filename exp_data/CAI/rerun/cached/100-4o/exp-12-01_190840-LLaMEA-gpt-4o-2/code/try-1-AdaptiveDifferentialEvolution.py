import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        evaluations = 0

        def select_parents():
            return np.random.choice(self.pop_size, 3, replace=False)

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                a, b, c = select_parents()
                x_i = self.population[i]
                x_a, x_b, x_c = self.population[a], self.population[b], self.population[c]

                mutant_vector = x_a + self.F * (x_b - x_c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.copy(x_i)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == np.random.randint(0, self.dim):
                        trial_vector[j] = mutant_vector[j]

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector

                if trial_fitness < func(x_i):
                    self.population[i] = trial_vector

        return self.best_solution