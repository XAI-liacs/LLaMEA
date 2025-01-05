import numpy as np

class AdaptiveQuantumGenetic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(15, dim)
        self.mutation_rate = 0.1
        self.quantum_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(population[i]) for i in range(self.population_size)])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Select parents for crossover using tournament selection
            new_population = []
            for _ in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                parent1 = min(indices, key=lambda idx: scores[idx])
                parent2 = min(set(indices) - {parent1}, key=lambda idx: scores[idx])
                new_population.append(self._crossover(population[parent1], population[parent2], lb, ub))
            
            # Mutation and quantum-inspired mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    new_population[i] = self._mutate(new_population[i], lb, ub)
                if np.random.rand() < self.quantum_rate:
                    new_population[i] = self._quantum_mutate(new_population[i], population, scores, lb, ub)

            # Evaluate new population
            new_scores = np.array([func(individual) for individual in new_population])
            evaluations += self.population_size

            # Select the next generation
            combined_population = np.vstack((population, new_population))
            combined_scores = np.hstack((scores, new_scores))
            best_indices = np.argsort(combined_scores)[:self.population_size]
            population = combined_population[best_indices]
            scores = combined_scores[best_indices]

        best_index = np.argmin(scores)
        return population[best_index], scores[best_index]

    def _crossover(self, parent1, parent2, lb, ub):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.dim)
            child = alpha * parent1 + (1 - alpha) * parent2
            return np.clip(child, lb, ub)
        return parent1 if np.random.rand() > 0.5 else parent2

    def _mutate(self, individual, lb, ub):
        mutation_vector = np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
        return np.clip(individual + mutation_vector, lb, ub)

    def _quantum_mutate(self, individual, population, scores, lb, ub):
        best_individual = population[np.argmin(scores)]
        quantum_step = np.random.normal(loc=0, scale=0.1, size=self.dim) * (ub - lb)
        return np.clip(best_individual + quantum_step, lb, ub)