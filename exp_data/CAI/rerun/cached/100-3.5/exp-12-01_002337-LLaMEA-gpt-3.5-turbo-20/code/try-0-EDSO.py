import numpy as np

class EDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        def init_population():
            return np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))

        def differential_evolution(population, f=0.5, cr=0.7):
            new_population = np.copy(population)
            for i in range(self.pop_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < cr or j == j_rand:
                        new_population[i, j] = population[a, j] + f * (population[b, j] - population[c, j])
            return new_population

        def swarm_intelligence(population):
            best = population[np.argmin([func(ind) for ind in population])]
            for i in range(self.pop_size):
                population[i] = 0.9 * population[i] + 0.1 * best
            return population

        population = init_population()
        for _ in range(self.max_iter):
            population = differential_evolution(population)
            population = swarm_intelligence(population)

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution