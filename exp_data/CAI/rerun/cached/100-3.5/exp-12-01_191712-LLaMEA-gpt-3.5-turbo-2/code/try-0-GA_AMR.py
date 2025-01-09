import numpy as np

class GA_AMR:
    def __init__(self, budget, dim, pop_size=50, mutation_prob=0.01, elite_ratio=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.elite_ratio = elite_ratio

    def __call__(self, func):
        def mutate(parent, std_dev):
            return np.clip(parent + std_dev * np.random.randn(self.dim), -5.0, 5.0)

        def evaluate(population):
            return np.array([func(ind) for ind in population])

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = evaluate(population)
        elite_count = int(self.pop_size * self.elite_ratio)
        std_dev = 0.1

        for _ in range(self.budget):
            elite_idx = np.argsort(fitness)[:elite_count]
            elite_pop = population[elite_idx]

            offspring = [mutate(parent, std_dev) for parent in elite_pop]
            offspring_fitness = evaluate(offspring)

            population = np.vstack((population, offspring))
            fitness = np.concatenate((fitness, offspring_fitness))

            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:self.pop_size]]
            fitness = fitness[sorted_idx[:self.pop_size]]

            diversity = np.mean(np.std(population, axis=0))
            std_dev = std_dev * (1 + 0.01 * (diversity - 0.5))

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        return best_solution, best_fitness