import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50, inertia=0.5, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_values = np.full(self.swarm_size, np.inf)

        global_best_position = None
        global_best_value = np.inf

        evaluations = 0

        while evaluations < self.budget:
            # Dynamic inertia adjustment for better convergence
            self.inertia = 0.9 - 0.5 * (evaluations / self.budget) 

            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                value = func(swarm[i])
                evaluations += 1

                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = swarm[i].copy()

                if value < global_best_value:
                    global_best_value = value
                    global_best_position = swarm[i].copy()

            if global_best_value < self.f_opt:
                self.f_opt = global_best_value
                self.x_opt = global_best_position

            # Update velocity and positions
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                inertia_component = self.inertia * velocities[i]
                cognitive_component = self.cognitive * r1 * (personal_best_positions[i] - swarm[i])
                social_component = self.social * r2 * (global_best_position - swarm[i])
                velocities[i] = inertia_component + cognitive_component + social_component

                # Update position
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

        return self.f_opt, self.x_opt