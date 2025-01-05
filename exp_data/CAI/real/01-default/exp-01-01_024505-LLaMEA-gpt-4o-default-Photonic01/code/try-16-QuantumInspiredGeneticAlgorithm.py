import numpy as np

class QuantumInspiredGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1
        self.entanglement_factor = 0.5

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def select_parents(self, fitness):
        total_fitness = np.sum(1 / (fitness + 1e-9))
        selection_prob = (1 / (fitness + 1e-9)) / total_fitness
        indices = np.random.choice(self.pop_size, size=self.pop_size, p=selection_prob)
        parents = self.population[indices]
        return parents

    def crossover(self, parents):
        offspring = np.empty_like(parents)
        for i in range(0, self.pop_size, 2):
            if i+1 >= self.pop_size:
                break
            alpha = np.random.rand(self.dim)
            offspring[i] = alpha * parents[i] + (1 - alpha) * parents[i+1]
            offspring[i+1] = alpha * parents[i+1] + (1 - alpha) * parents[i]
        return offspring

    def mutate(self, offspring, lb, ub):
        mutation_matrix = np.random.rand(*offspring.shape) < self.mutation_rate
        random_values = lb + (ub - lb) * np.random.rand(*offspring.shape)
        offspring = np.where(mutation_matrix, random_values, offspring)
        return np.clip(offspring, lb, ub)

    def apply_quantum_entanglement(self, offspring):
        for i in range(self.pop_size):
            if np.random.rand() < self.entanglement_factor:
                partner_idx = np.random.randint(self.pop_size)
                qubit_superposition = 0.5 * (offspring[i] + offspring[partner_idx])
                offspring[i] = qubit_superposition + np.random.normal(0, 0.1, self.dim)
        return offspring

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += self.pop_size

            if evaluations >= self.budget:
                break

            parents = self.select_parents(fitness)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring, lb, ub)
            offspring = self.apply_quantum_entanglement(offspring)

            self.population = offspring

        return self.best_solution, self.best_fitness