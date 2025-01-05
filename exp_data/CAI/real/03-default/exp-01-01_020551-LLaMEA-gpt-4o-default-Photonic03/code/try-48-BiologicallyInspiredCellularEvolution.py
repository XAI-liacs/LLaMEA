import numpy as np

class BiologicallyInspiredCellularEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.grid_size = int(np.sqrt(self.population_size)) ** 2
        self.local_interaction_radius = 1
        self.replenish_rate = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.grid_size, self.dim))
        scores = np.array([func(individual) for individual in population])
        evaluations = self.grid_size

        while evaluations < self.budget:
            new_population = population.copy()
            for index in range(self.grid_size):
                local_best = self._get_local_best(population, scores, index)
                offspring = self._crossover(population[index], local_best, ub, lb)
                offspring = self._mutate(offspring, ub, lb)
                offspring_score = func(offspring)
                evaluations += 1

                if offspring_score < scores[index]:
                    new_population[index] = offspring
                    scores[index] = offspring_score

                # Apply replenishment strategy
                if np.random.rand() < self.replenish_rate:
                    new_population[index] = np.random.uniform(lb, ub, self.dim)
                    scores[index] = func(new_population[index])
                    evaluations += 1

                if evaluations >= self.budget:
                    break

            population = new_population

        best_index = np.argmin(scores)
        return population[best_index], scores[best_index]

    def _crossover(self, parent1, parent2, ub, lb):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim - 1)
            offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return np.clip(offspring, lb, ub)
        return parent1

    def _mutate(self, individual, ub, lb):
        mutation_vector = np.random.normal(0, self.mutation_rate, self.dim) * (ub - lb)
        return np.clip(individual + mutation_vector, lb, ub)

    def _get_local_best(self, population, scores, index):
        row_size = int(np.sqrt(self.grid_size))
        row, col = divmod(index, row_size)
        local_indices = [(row + i, col + j) for i in range(-self.local_interaction_radius, self.local_interaction_radius + 1)
                         for j in range(-self.local_interaction_radius, self.local_interaction_radius + 1)
                         if 0 <= row + i < row_size and 0 <= col + j < row_size]
        local_indices = [i * row_size + j for (i, j) in local_indices]
        local_scores = scores[local_indices]
        best_local_index = local_indices[np.argmin(local_scores)]
        return population[best_local_index]