import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        budget_used = self.pop_size

        while budget_used < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                budget_used += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best = trial
                        best_idx = i

                if budget_used >= self.budget:
                    break

            # Neighborhood Search
            if budget_used < self.budget:
                for neighbor in self._get_neighborhood(best):
                    neighbor_fitness = func(neighbor)
                    budget_used += 1
                    if neighbor_fitness < fitness[best_idx]:
                        best = neighbor
                        best_idx = np.argmin(fitness)
                        fitness[best_idx] = neighbor_fitness
                        population[best_idx] = neighbor

                    if budget_used >= self.budget:
                        break

        return best

    def _get_neighborhood(self, best):
        neighborhood_size = self.dim
        for _ in range(neighborhood_size):
            neighbor = best + np.random.normal(0, 0.1, self.dim)
            yield np.clip(neighbor, self.lower_bound, self.upper_bound)