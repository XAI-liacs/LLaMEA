import numpy as np

class QuantumEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.alpha = 0.1  # rotation angle for quantum gate

    def initialize_population(self, bounds):
        quantum_states = np.random.rand(self.population_size, self.dim, 2)  # amplitude for |0> and |1>
        quantum_states /= np.linalg.norm(quantum_states, axis=2, keepdims=True)
        return quantum_states, self.measure_population(quantum_states, bounds)

    def measure_population(self, quantum_states, bounds):
        binary_population = np.argmax(quantum_states, axis=2)
        decimal_population = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (binary_population / (2**self.dim - 1))
        return decimal_population

    def quantum_rotation(self, quantum_states, best_individual):
        for i in range(self.population_size):
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    if best_individual[j] == 0:
                        quantum_states[i, j, 0], quantum_states[i, j, 1] = \
                            np.cos(self.alpha) * quantum_states[i, j, 0] - np.sin(self.alpha) * quantum_states[i, j, 1], \
                            np.sin(self.alpha) * quantum_states[i, j, 0] + np.cos(self.alpha) * quantum_states[i, j, 1]
                    else:
                        quantum_states[i, j, 0], quantum_states[i, j, 1] = \
                            np.cos(self.alpha) * quantum_states[i, j, 0] + np.sin(self.alpha) * quantum_states[i, j, 1], \
                            -np.sin(self.alpha) * quantum_states[i, j, 0] + np.cos(self.alpha) * quantum_states[i, j, 1]
        return quantum_states

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        quantum_states, population = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            quantum_states = self.quantum_rotation(quantum_states, best_individual)
            population = self.measure_population(quantum_states, bounds)
            
            for i in range(self.population_size):
                fitness[i] = func(population[i])
                eval_count += 1
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_individual = population[i]
                    
                if eval_count >= self.budget:
                    break

        return best_individual