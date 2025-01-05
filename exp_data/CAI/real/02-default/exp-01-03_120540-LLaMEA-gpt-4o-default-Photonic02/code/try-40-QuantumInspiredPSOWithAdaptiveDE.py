import numpy as np

class QuantumInspiredPSOWithAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.729
        self.cognitive_coefficient = 1.49445
        self.social_coefficient = 1.49445
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.velocities = None
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = None

    def initialize_positions_and_velocities(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(lb - ub, ub - lb, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))

    def quantum_inspired_mutation(self, individual):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        mutated_individual = individual + self.mutation_factor * quantum_flip
        return mutated_individual

    def adaptive_differential_evolution(self, target_idx, evaluations, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)

        mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        trial_vector = np.copy(self.positions[target_idx])

        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector[crossover_mask] = mutant_vector[crossover_mask]
        trial_vector = np.clip(trial_vector, lb, ub)

        return trial_vector, func(trial_vector)

    def __call__(self, func):
        self.initialize_positions_and_velocities(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # PSO velocity and position update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.cognitive_coefficient * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                                      self.social_coefficient * r2 * (self.best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                current_score = func(self.positions[i])
                evaluations += 1

                # Update personal best
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = np.copy(self.positions[i])

                # Update global best
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = np.copy(self.positions[i])

                # Adaptive DE refinement
                trial_vector, trial_score = self.adaptive_differential_evolution(i, evaluations, func)
                evaluations += 1
                if trial_score < current_score:
                    self.positions[i] = trial_vector
                    current_score = trial_score

                # Adjust mutation factor
                self.mutation_factor = 0.5 * (1 - evaluations / self.budget)

        # Quantum-inspired global mutation for final exploration
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            mutated_position = self.quantum_inspired_mutation(self.positions[i])
            mutated_position = np.clip(mutated_position, func.bounds.lb, func.bounds.ub)
            mutated_score = func(mutated_position)
            evaluations += 1
            if mutated_score < self.best_score:
                self.best_score = mutated_score
                self.best_position = mutated_position