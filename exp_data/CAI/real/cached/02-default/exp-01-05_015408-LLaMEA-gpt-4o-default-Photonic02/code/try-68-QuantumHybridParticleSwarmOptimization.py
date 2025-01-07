import numpy as np

class QuantumHybridParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(position_population)
        personal_best_fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size

        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity
                inertia = self.inertia_weight * velocities[i]
                cognitive_component = self.cognitive_coef * np.random.rand(self.dim) * (personal_best_positions[i] - position_population[i])
                social_component = self.social_coef * np.random.rand(self.dim) * (global_best_position - position_population[i])
                velocities[i] = inertia + cognitive_component + social_component

                # Update position
                position_population[i] += velocities[i]
                position_population[i] = np.clip(position_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness
                    personal_best_positions[i] = position_population[i]

                # Update global best
                if new_fitness < personal_best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_fitness[global_best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position