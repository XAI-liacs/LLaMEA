import numpy as np

class DynamicQuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.entanglement_factor = 0.7
        self.mutation_rate = 0.1
        self.mutation_adaptation_rate = 0.03
        self.population = np.random.rand(self.population_size, self.dim)
        self.population *= (1 + 1) - 1

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _select_parents(self, fitness):
        idx = np.argsort(fitness)
        return self.population[idx][:self.population_size // 2]

    def _crossover(self, parents):
        offspring = []
        for _ in range(self.population_size - len(parents)):
            if np.random.rand() < self.entanglement_factor:
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                cross_point = np.random.randint(0, self.dim)
                child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                offspring.append(child)
        return np.array(offspring)

    def _mutate(self, population, lb, ub):
        for individual in population:
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 1, self.dim)
                individual += self.mutation_adaptation_rate * mutation_vector
                np.clip(individual, lb, ub, out=individual)
        return population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = lb + (ub - lb) * self.population

        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.budget):
            fitness = self._evaluate_population(func)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = self.population[best_idx]

            parents = self._select_parents(fitness)
            offspring = self._crossover(parents)
            self.population = np.vstack((parents, offspring))
            self.population = self._mutate(self.population, lb, ub)

        return best_solution