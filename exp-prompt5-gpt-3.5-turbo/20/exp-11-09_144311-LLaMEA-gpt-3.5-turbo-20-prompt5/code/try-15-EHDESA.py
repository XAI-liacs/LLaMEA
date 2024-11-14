import numpy as np

class EHDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.min_cr = 0.5
        self.max_cr = 0.9
        self.min_f = 0.3
        self.max_f = 0.8
        self.alpha = 0.9
        self.sigma = 0.1

    def __call__(self, func):
        def de_mutate(population, target_idx, cr, f):
            candidates = population[np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)]
            donor_vector = population[target_idx] + f * (candidates[0] - candidates[1])
            for i in range(self.dim):
                if np.random.rand() > cr:
                    donor_vector[i] = population[target_idx][i]
            return donor_vector

        def sa_mutation(candidate, best, t):
            return candidate + self.sigma * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        t = 0

        while t < self.budget:
            new_population = np.zeros_like(population)
            cr = self.min_cr + (self.max_cr - self.min_cr) * t / self.budget
            f = self.max_f - (self.max_f - self.min_f) * t / self.budget
            for i in range(self.pop_size):
                candidate = de_mutate(population, i, cr, f)
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < fitness[best_idx]:
                        best_solution = candidate
                        best_idx = i
                else:
                    new_population[i] = sa_mutation(population[i], best_solution, t)
                t += 1

            population = new_population

        return best_solution