import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha_init = 0.5
        self.beta_init = 0.5
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        alpha = self.alpha_init
        beta = self.beta_init

        while evaluations < self.budget:
            diversity = self.calculate_diversity(position_population)
            if diversity < self.diversity_threshold:
                alpha, beta = self.adapt_parameters(alpha, beta, increase=True)
            else:
                alpha, beta = self.adapt_parameters(alpha, beta, increase=False)

            for i in range(self.population_size):
                if np.random.rand() < alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], beta)
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
        delta_theta = beta * (best_quantum_bits - quantum_bits) + (np.random.rand(self.dim) - 0.5) * 0.1
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def adapt_parameters(self, alpha, beta, increase):
        factor = 0.1
        if increase:
            alpha = min(1.0, alpha + factor)
            beta = min(1.0, beta + factor)
        else:
            alpha = max(0.0, alpha - factor)
            beta = max(0.0, beta - factor)
        return alpha, beta

    def calculate_diversity(self, population):
        mean_position = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - mean_position, axis=1))
        return diversity