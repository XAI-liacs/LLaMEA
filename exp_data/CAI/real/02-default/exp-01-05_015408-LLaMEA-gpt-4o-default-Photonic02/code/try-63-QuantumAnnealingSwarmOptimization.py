import numpy as np

class QuantumAnnealingSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents(fitness)
                child = self.crossover(quantum_population[parent1], quantum_population[parent2])
                child = self.mutate(child)
                new_position = self.quantum_to_position(child, lb, ub)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < fitness[i]:
                    new_population.append((child, new_fitness))
                else:
                    acceptance_prob = np.exp((fitness[i] - new_fitness) / (self.temperature + 1e-9))
                    if np.random.rand() < acceptance_prob:
                        new_population.append((child, new_fitness))
                    else:
                        new_population.append((quantum_population[i], fitness[i]))

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = new_position

                if evaluations >= self.budget:
                    break

            quantum_population, fitness = zip(*new_population)
            quantum_population = np.array(quantum_population)
            fitness = np.array(fitness)
            self.temperature *= self.cooling_rate

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def select_parents(self, fitness):
        probabilities = 1 / (fitness - np.min(fitness) + 1e-9)
        probabilities /= probabilities.sum()
        return np.random.choice(range(self.population_size), size=2, p=probabilities, replace=False)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2
        return parent1

    def mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, 0.1, size=self.dim)
            child = np.clip(child + mutation_vector, 0, 1)
        return child