import numpy as np

class ImprovedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.elitism_rate = 0.2
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        elite_size = int(self.elitism_rate * self.population_size)
        
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        
        while evaluations < self.budget:
            # Preserve elite members
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_quantum = quantum_population[elite_indices]
            
            for i in range(self.population_size):
                if i not in elite_indices:
                    # Adaptive quantum rotation gate with diversity preservation
                    if np.random.rand() < self.alpha:
                        quantum_population[i] = self.update_quantum_bits(
                            quantum_population[i], quantum_population[best_index], fitness, i)
                    
                    # Convert quantum representation to classical position
                    position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                    
                    # Evaluate new position
                    new_fitness = func(position_population[i])
                    evaluations += 1
                    
                    # Selection: keep the better solution
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness

                    # Update best position
                    if new_fitness < fitness[best_index]:
                        best_index = i
                        best_position = position_population[i]
                    
                    # Diversity enhancement
                    if np.std(position_population, axis=0).mean() < self.diversity_threshold:
                        quantum_population[i] = np.random.rand(self.dim)

                if evaluations >= self.budget:
                    break

            # Reinsert elite solutions
            quantum_population[:elite_size] = elite_quantum

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, fitness, index):
        # Adaptive quantum rotation inspired update
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits