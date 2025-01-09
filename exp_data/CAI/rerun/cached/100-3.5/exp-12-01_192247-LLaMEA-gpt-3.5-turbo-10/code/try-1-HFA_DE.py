import numpy as np

class HFA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def levy_flight(dim):
            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, dim)
            v = np.random.normal(0, 1, dim)
            step = u / np.power(np.abs(v), 1/beta)
            return step

        def de_mutate(x, a, b, c, F):
            return x + F * (a - x) + F * (b - c)

        def clip_bounds(x):
            return np.clip(x, -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = [func(ind) for ind in population]

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]

        while self.budget > 0:
            for i in range(self.budget):
                r = np.random.uniform(0, 1)
                if r < 0.7:
                    for j in range(self.dim):
                        r1, r2, r3 = np.random.choice(population, 3, replace=False)
                        new_sol = de_mutate(population[i, j], r1[j], r2[j], r3[j], 0.5)
                        population[i, j] = clip_bounds(new_sol)
                else:
                    step = levy_flight(self.dim)
                    population[i] += step
                    population[i] = clip_bounds(population[i])

                fitness_values[i] = func(population[i])
                if fitness_values[i] < func(best_solution):
                    best_solution = np.copy(population[i])

                self.budget -= 1
                if self.budget <= 0:
                    break

        return best_solution