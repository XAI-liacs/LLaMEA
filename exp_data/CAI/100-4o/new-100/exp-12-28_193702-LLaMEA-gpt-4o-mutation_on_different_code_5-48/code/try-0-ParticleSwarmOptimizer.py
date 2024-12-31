import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia_weight=0.9, cognitive_coeff=2.0, social_coeff=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.f_opt = np.Inf
        self.x_opt = None
        self.v_max = 0.2 * (5.0 - (-5.0))

    def __call__(self, func):
        # Initialize particle positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        num_evaluations = 0

        while num_evaluations < self.budget:
            for i in range(self.swarm_size):
                if num_evaluations >= self.budget:
                    break
                
                score = func(positions[i])
                num_evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * r2 * (self.x_opt - positions[i])
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_component + social_component)

                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)  # clip velocities
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)  # ensure particles stay within bounds

            # Dynamically adjust inertia weight
            self.inertia_weight *= 0.99

        return self.f_opt, self.x_opt