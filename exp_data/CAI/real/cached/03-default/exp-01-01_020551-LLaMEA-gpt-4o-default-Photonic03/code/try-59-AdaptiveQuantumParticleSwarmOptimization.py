import numpy as np

class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(30, 5 * dim)
        self.initial_cognitive_weight = 1.5
        self.initial_social_weight = 1.5
        self.quantum_factor = 0.1
        self.adaptive_rate = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim)) * (ub - lb)
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.num_particles
        cognitive_weight = self.initial_cognitive_weight
        social_weight = self.initial_social_weight

        while evaluations < self.budget:
            for i in range(self.num_particles):
                velocities[i] += cognitive_weight * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                velocities[i] += social_weight * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = np.clip(velocities[i], -abs(ub - lb), abs(ub - lb))

                # Quantum-inspired position update
                if np.random.rand() < self.quantum_factor:
                    q = np.random.normal(0, 0.1, self.dim)
                    positions[i] = global_best_position + q * (ub - lb)
                else:
                    positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], lb, ub)
                score = func(positions[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                if score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = personal_best_positions[global_best_index].copy()

            # Adaptive parameter tuning
            cognitive_weight = self.initial_cognitive_weight - self.adaptive_rate * (evaluations / self.budget)
            social_weight = self.initial_social_weight + self.adaptive_rate * (evaluations / self.budget)

        return global_best_position, personal_best_scores[global_best_index]