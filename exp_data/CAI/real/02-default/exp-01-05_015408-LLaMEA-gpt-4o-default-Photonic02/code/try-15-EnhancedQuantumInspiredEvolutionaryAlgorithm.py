import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta_initial = 0.5
        self.beta_final = 0.1
        self.elite_fraction = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            beta = self.update_beta(evaluations)
            elite_indices = self.get_elite_indices(fitness)
            
            for i in range(self.population_size):
                if i in elite_indices:  # Retain elite individuals
                    continue
                if np.random.rand() < self.alpha:
                    best_quantum_bits = quantum_population[best_index]
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], best_quantum_bits, beta)

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, beta):
        delta_theta = beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def update_beta(self, evaluations):
        progress = evaluations / self.budget
        return self.beta_initial + progress * (self.beta_final - self.beta_initial)

    def get_elite_indices(self, fitness):
        num_elites = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(fitness)[:num_elites]
        return elite_indices