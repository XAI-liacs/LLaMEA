import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.quantum_entropy_factor = 0.1
        self.global_attraction_weight = 0.9
        self.personal_attraction_weight = 1.4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particle_positions = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        particle_velocities = np.random.rand(self.population_size, self.dim) * 0.1 * (ub - lb)
        personal_best_positions = np.copy(particle_positions)
        personal_best_scores = np.array([func(ind) for ind in particle_positions])
        evaluations = self.population_size
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update particle velocity with quantum-inspired dynamics
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.personal_attraction_weight * r1 * (personal_best_positions[i] - particle_positions[i])
                social_component = self.global_attraction_weight * r2 * (global_best_position - particle_positions[i])
                quantum_component = self.quantum_entropy_factor * np.random.randn(self.dim) * (ub - lb)
                
                particle_velocities[i] = 0.5 * particle_velocities[i] + cognitive_component + social_component + quantum_component
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], lb, ub)

                # Evaluate new position
                current_score = func(particle_positions[i])
                evaluations += 1

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = particle_positions[i]

                # Update global best
                if current_score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = particle_positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_scores[global_best_index]