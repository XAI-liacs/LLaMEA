import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.rotation_angle = 0.1
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_pop = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = self.quantum_to_real(quantum_pop, lb, ub)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_quantum_pop = np.zeros_like(quantum_pop)

            for i in range(self.population_size):
                # Select partner for quantum rotation
                partner_indices = list(range(self.population_size))
                partner_indices.remove(i)
                partner_idx = np.random.choice(partner_indices)
                partner = quantum_pop[partner_idx]

                # Calculate rotation direction
                direction = np.sign(partner - quantum_pop[i])

                # Apply quantum rotation
                next_quantum_pop[i] = quantum_pop[i] + direction * self.rotation_angle
                next_quantum_pop[i] = np.clip(next_quantum_pop[i], 0, 1)

            # Convert quantum population to real values
            next_pop = self.quantum_to_real(next_quantum_pop, lb, ub)
            next_fitness = np.array([func(x) for x in next_pop])
            evaluations += self.population_size

            # Select survivors
            for i in range(self.population_size):
                if next_fitness[i] < fitness[i]:
                    fitness[i] = next_fitness[i]
                    quantum_pop[i] = next_quantum_pop[i]
                    if next_fitness[i] < fitness[best_idx]:
                        best_idx = i
                        best_global = next_pop[i]

            self.history.append(best_global)

        return best_global

    def quantum_to_real(self, quantum_pop, lb, ub):
        scale = ub - lb
        return lb + quantum_pop * scale