import numpy as np

class DynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim * 2)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.velocity_clamp = 0.1  # Velocity clamping factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, float('inf'))

        global_best_position = None
        global_best_score = float('inf')

        eval_count = 0
        while eval_count < self.budget:
            # Evaluate swarm
            for i in range(self.population_size):
                score = func(swarm[i])
                eval_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(swarm[i])
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = np.copy(swarm[i])
                if eval_count >= self.budget:
                    break

            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - swarm) +
                          self.social_coeff * r2 * (global_best_position - swarm))
            velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

        return global_best_position, global_best_score