import numpy as np

class AdaptiveDiffEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # scaling factor
        self.CR = 0.9  # crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Mutation and recombination
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluation
                trial_fitness = func(trial)
                self.eval_count += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                elif np.random.rand() < 0.1:  # Random walk exploration with a small probability
                    self.population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1

        best_index = np.argmin(self.fitness)
        return self.population[best_index]

# Example usage:
# optimizer = AdaptiveDiffEvolution(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)