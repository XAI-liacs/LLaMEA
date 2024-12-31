import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia_weight=0.9, cognitive_coefficient=2.0, social_coefficient=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.f_opt = np.Inf
        self.x_opt = None
        self.velocity_clamp = 0.1 * (5.0 - -5.0)  # 10% of the range

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(x) for x in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coefficient * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.social_coefficient * r2 * (global_best_position - swarm[i]))
                
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)

                score = func(swarm[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

                if evaluations >= self.budget:
                    break

            self.inertia_weight *= 0.99  # Adaptive inertia weight decay
            self.cognitive_coefficient *= 1.01  # Dynamic cognitive coefficient
            self.social_coefficient *= 0.99  # Dynamic social coefficient

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt