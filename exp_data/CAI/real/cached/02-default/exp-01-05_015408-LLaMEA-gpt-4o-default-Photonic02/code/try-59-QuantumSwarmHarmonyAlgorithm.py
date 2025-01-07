import numpy as np

class QuantumSwarmHarmonyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.harmony_memory_size = 5 * dim
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.3   # Pitch Adjusting Rate
        self.adap_factor = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        
        harmony_memory = position_population[:self.harmony_memory_size]
        harmony_fitness = fitness[:self.harmony_memory_size]
        best_index = np.argmin(harmony_fitness)
        best_position = harmony_memory[best_index]

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_index = np.random.randint(0, self.harmony_memory_size)
                    new_harmony[d] = harmony_memory[selected_index, d]
                    if np.random.rand() < self.par:
                        new_harmony[d] += np.random.uniform(-self.adap_factor, self.adap_factor) * (ub[d] - lb[d])
                else:
                    new_harmony[d] = np.random.uniform(lb[d], ub[d])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < np.max(harmony_fitness):
                worst_index = np.argmax(harmony_fitness)
                harmony_memory[worst_index] = new_harmony
                harmony_fitness[worst_index] = new_fitness

            if new_fitness < harmony_fitness[best_index]:
                best_index = np.argmin(harmony_fitness)
                best_position = harmony_memory[best_index]

            if evaluations >= self.budget:
                break

        return best_position, harmony_fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position