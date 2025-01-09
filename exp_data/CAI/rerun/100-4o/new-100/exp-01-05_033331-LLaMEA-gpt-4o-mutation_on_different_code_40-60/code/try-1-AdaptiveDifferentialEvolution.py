import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def compute_crowding_distance(self, population, fitness):
        distances = np.zeros(self.initial_pop_size)
        for i in range(self.dim):
            sorted_indices = np.argsort(population[:, i])
            sorted_fitness = fitness[sorted_indices]
            distances[sorted_indices[0]] = np.Inf
            distances[sorted_indices[-1]] = np.Inf
            for j in range(1, self.initial_pop_size - 1):
                distances[sorted_indices[j]] += (sorted_fitness[j+1] - sorted_fitness[j-1])
        return distances 

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.initial_pop_size

        while evals < self.budget:
            crowding_distances = self.compute_crowding_distance(population, fitness)
            for i in range(self.initial_pop_size):
                if evals >= self.budget:
                    break

                indices = list(range(self.initial_pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            self.F = 0.5 + 0.5 * np.random.rand()  # Adaptive mutation factor
            self.CR = 0.8 + 0.2 * np.random.rand()  # Adaptive crossover rate

            if evals < self.budget / 2:
                self.initial_pop_size = max(10, int(self.initial_pop_size * 0.9))

        return self.f_opt, self.x_opt