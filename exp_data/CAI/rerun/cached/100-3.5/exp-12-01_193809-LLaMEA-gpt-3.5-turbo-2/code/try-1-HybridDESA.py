import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def de_mutate(population, F):
            idxs = np.random.choice(len(population), 3, replace=False)
            a, b, c = population[idxs]
            return a + F * (b - c)

        def bounded_mutation(mutant, lower, upper):
            return np.clip(mutant, lower, upper)

        def sa_acceptance(current_val, new_val, T):
            if new_val < current_val:
                return True
            return np.random.rand() < np.exp((current_val - new_val) / T)

        def sa_cooling_schedule(T, cooling_rate):
            return T * cooling_rate

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()

        F = 0.5
        T = 0.1
        final_T = 0.0001
        cooling_rate = 0.99

        while self.budget > 0:
            for i in range(len(population)):
                mutant = de_mutate(population, F)
                mutant = bounded_mutation(mutant, -5.0, 5.0)
                new_fitness = func(mutant)

                if sa_acceptance(fitness[i], new_fitness, T):
                    population[i] = mutant
                    fitness[i] = new_fitness

                    if new_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = population[i].copy()

                self.budget -= 1
                T = sa_cooling_schedule(T, cooling_rate)
                if T < final_T or self.budget == 0:
                    break

        return best_solution