import numpy as np

class QuantumGeneticSwarmAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.quantum_bits = 2 * dim
        self.alpha = 0.5  # Quantum update rate
        self.beta = 0.9   # Inertia weight for PSO
        self.c1, self.c2 = 2.0, 2.0  # PSO coefficients

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.quantum_bits)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        velocities = np.zeros_like(position_population)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        global_best_position = position_population[best_index]
        personal_best_positions = position_population.copy()
        personal_best_fitness = fitness.copy()

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity and position using PSO dynamics
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.beta * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - position_population[i]) +
                                 self.c2 * r2 * (global_best_position - position_population[i]))
                position_population[i] += velocities[i]
                position_population[i] = np.clip(position_population[i], lb, ub)

                # Quantum bit rotation inspired update
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.quantum_update(quantum_population[i], global_best_position)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = position_population[i]

                if new_fitness < personal_best_fitness[best_index]:
                    best_index = i
                    global_best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits[:, :self.dim] * (ub - lb)
        return position

    def quantum_update(self, quantum_bits, global_best_position):
        # Quantum rotation and genetic mutation inspired update
        delta_theta = self.alpha * (global_best_position - quantum_bits[:self.dim])
        new_quantum_bits = quantum_bits[:self.dim] + delta_theta
        mutation = np.random.normal(0, 0.1, size=self.dim)
        new_quantum_bits = new_quantum_bits + mutation
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        quantum_bits[:self.dim] = new_quantum_bits
        return quantum_bits