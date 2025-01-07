import numpy as np

class BioInspiredDynamicParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.inertia_decay = 0.99
        self.min_inertia = 0.4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.swarm_size, self.dim))
        personal_best_positions = swarm_positions.copy()
        personal_best_scores = np.array([func(p) for p in swarm_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(2, self.dim)
                swarm_velocities[i] = (
                    self.inertia_weight * swarm_velocities[i] +
                    self.cognitive_coeff * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                    self.social_coeff * r2 * (global_best_position - swarm_positions[i])
                )
                # Update position
                swarm_positions[i] += swarm_velocities[i]
                swarm_positions[i] = np.clip(swarm_positions[i], lb, ub)

                # Evaluate new position
                score = func(swarm_positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm_positions[i].copy()

                # Update global best
                if score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = personal_best_positions[i].copy()

            # Decay inertia
            self.inertia_weight = max(self.min_inertia, self.inertia_weight * self.inertia_decay)

        return global_best_position, personal_best_scores[global_best_index]