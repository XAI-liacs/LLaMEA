import numpy as np

class HSDEH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.hmcr = 0.7
        self.par = 0.5
        self.f = 0.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def harmonize(population):
            new_population = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_population[i, j] = population[np.random.randint(self.pop_size), j]
                    else:
                        new_population[i, j] = np.random.uniform(-5.0, 5.0)
            return new_population

        def evolve(population):
            new_population = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                new_individual = population[i] + self.f * (a - b)
                for j in range(self.dim):
                    if np.random.rand() > self.par:
                        new_individual[j] = c[j]
                new_population[i] = np.clip(new_individual, -5.0, 5.0)
            return new_population

        population = initialize_population()
        evaluations_left = self.budget
        while evaluations_left > 0:
            harmonized_population = harmonize(population)
            mutated_population = evolve(harmonized_population)
            population_evaluations = evaluate_population(population)
            mutated_evaluations = evaluate_population(mutated_population)
            for i in range(self.pop_size):
                if mutated_evaluations[i] < population_evaluations[i]:
                    population[i] = mutated_population[i]
            evaluations_left -= self.pop_size

        best_idx = np.argmin(evaluate_population(population))
        return population[best_idx]