import numpy as np

class BeeColonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_bees = 20 * dim
        self.num_elite_bees = int(0.1 * self.num_bees)
        self.num_scout_bees = int(0.2 * self.num_bees)
        self.waggle_factor = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.num_bees, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.num_bees
        best_index = np.argmin(fitness)
        best_position = population[best_index]

        while evaluations < self.budget:
            # Recruit elite bees
            elite_indices = np.argsort(fitness)[:self.num_elite_bees]
            for idx in elite_indices:
                new_position = self.explore_around(population[idx], lb, ub)
                new_fitness = func(new_position)
                evaluations += 1
                if new_fitness < fitness[idx]:
                    fitness[idx] = new_fitness
                    population[idx] = new_position
                    if new_fitness < fitness[best_index]:
                        best_index = idx
                        best_position = new_position

                if evaluations >= self.budget:
                    break

            # Recruit scout bees
            for _ in range(self.num_scout_bees):
                scout_idx = np.random.choice(self.num_bees)
                new_position = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_position)
                evaluations += 1
                if new_fitness < fitness[scout_idx]:
                    fitness[scout_idx] = new_fitness
                    population[scout_idx] = new_position
                    if new_fitness < fitness[best_index]:
                        best_index = scout_idx
                        best_position = new_position

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def explore_around(self, position, lb, ub):
        # Explore new position around the current position
        new_position = position + self.waggle_factor * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = np.clip(new_position, lb, ub)
        return new_position