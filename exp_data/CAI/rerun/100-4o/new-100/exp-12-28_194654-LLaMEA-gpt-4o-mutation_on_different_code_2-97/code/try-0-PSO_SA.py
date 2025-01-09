import numpy as np

class PSO_SA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.swarm_size = 50
        self.inertia = 0.7
        self.cognitive = 2.0
        self.social = 2.0
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive * r1 * (personal_best_positions[i] - swarm[i])
                social_component = self.social * r2 * (global_best_position - swarm[i])
                velocities[i] = (self.inertia * velocities[i] + cognitive_component + social_component)
                
                # Update position
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)
                
                # Evaluate function
                f = func(swarm[i])
                evaluations += 1

                # Check for personal best update
                if f < personal_best_scores[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_scores[i] = f

                    # Check for global best update with SA acceptance
                    if f < global_best_score or np.random.rand() < np.exp((global_best_score - f) / self.temperature):
                        global_best_position = swarm[i]
                        global_best_score = f

                    # Update temperature
                    self.temperature *= self.cooling_rate

            if evaluations >= self.budget:
                break

        self.f_opt, self.x_opt = global_best_score, global_best_position
        return self.f_opt, self.x_opt