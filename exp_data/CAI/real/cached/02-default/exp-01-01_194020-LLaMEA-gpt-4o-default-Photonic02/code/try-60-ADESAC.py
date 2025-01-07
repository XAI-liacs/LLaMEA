import numpy as np

class ADESAC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.5  # initial differential weight
        self.CR = 0.9  # initial crossover probability

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def mutate(self, idx, population, F):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = a + F * (b - c)
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        self.lb, self.ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = self.initialize_population(self.lb, self.ub)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                F = 0.4 + 0.3 * np.random.rand()  # self-adaptive F
                CR = 0.8 + 0.2 * np.random.rand()  # self-adaptive CR
                
                mutant = self.mutate(i, population, F)
                trial = self.crossover(population[i], mutant, CR)
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]