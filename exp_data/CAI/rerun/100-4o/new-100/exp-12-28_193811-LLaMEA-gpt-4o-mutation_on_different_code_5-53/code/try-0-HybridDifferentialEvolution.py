import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def mutate(self, population, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = population[a] + self.f * (population[b] - population[c])
        return np.clip(mutant_vector, -5.0, 5.0)

    def crossover(self, target, mutant):
        crossover_vector = np.where(np.random.rand(self.dim) < self.cr, mutant, target)
        return crossover_vector

    def local_search(self, x, func):
        best_x = x
        best_f = func(x)
        for _ in range(3):  # limited number of local search steps
            neighbor = x + np.random.normal(0, 0.1, self.dim)
            neighbor = np.clip(neighbor, -5.0, 5.0)
            f = func(neighbor)
            if f < best_f:
                best_x, best_f = neighbor, f
        return best_x, best_f

    def __call__(self, func):
        population = self.initialize_population(func.bounds)
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        for _ in range(self.budget - evaluations):
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.f_opt:
                    local_x, local_f = self.local_search(trial, func)
                    if local_f < self.f_opt:
                        self.f_opt, self.x_opt = local_f, local_x

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt