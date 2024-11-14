import numpy as np

class EHDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.9
        self.f = 0.8
        self.alpha = 0.9
        self.sigma = 0.1
        self.f_min = 0.2
        self.f_max = 0.9
        self.cr_min = 0.2
        self.cr_max = 0.9

    def __call__(self, func):
        def de_mutate(population, target_idx, f):
            candidates = population[np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)]
            donor_vector = population[target_idx] + f * (candidates[0] - candidates[1])
            for i in range(self.dim):
                if np.random.rand() > self.cr:
                    donor_vector[i] = population[target_idx][i]
            return donor_vector

        def sa_mutation(candidate, best, t, sigma):
            return candidate + sigma * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        t = 0

        while t < self.budget:
            new_population = np.zeros_like(population)
            f = self.f_min + (self.f_max - self.f_min) * t / self.budget
            cr = self.cr_min + (self.cr_max - self.cr_min) * t / self.budget
            for i in range(self.pop_size):
                candidate = de_mutate(population, i, f)
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < fitness[best_idx]:
                        best_solution = candidate
                        best_idx = i
                else:
                    sigma = np.clip(self.sigma * np.exp(-t * self.alpha), 0.001, 0.5)
                    new_population[i] = sa_mutation(population[i], best_solution, t, sigma)
                t += 1

            population = new_population

        return best_solution