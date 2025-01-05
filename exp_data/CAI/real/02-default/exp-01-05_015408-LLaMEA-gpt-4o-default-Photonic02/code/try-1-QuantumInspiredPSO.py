import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_amplitude = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_coeff * r2 * (global_best_position - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                # Quantum-inspired update using quantum superposition
                quantum_position = population[i] + self.quantum_amplitude * np.random.normal(0, 1, self.dim)
                quantum_position = np.clip(quantum_position, lb, ub)
                quantum_fitness = func(quantum_position)
                evaluations += 1

                if quantum_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = quantum_position
                    personal_best_scores[i] = quantum_fitness

                # Update personal and global bests
                if personal_best_scores[i] < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = personal_best_positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_scores[global_best_index]