import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.5, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social * r2 * (global_best_position - positions[i])
                velocities[i] = (self.inertia * ((self.budget - evaluations) / self.budget)) * velocities[i] + cognitive_velocity + social_velocity
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt