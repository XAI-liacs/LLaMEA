import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.velocity_limit = 0.2 * (5 - (-5))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(lb-ub, ub-lb, (self.swarm_size, self.dim)) * self.velocity_limit  # Adjusted initialization
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.Inf)
        global_best_position = None
        global_best_value = np.Inf
        
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                fitness = func(positions[i])
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best_values[i] = fitness
                    personal_best_positions[i] = positions[i]

                if fitness < global_best_value:
                    global_best_value = fitness
                    global_best_position = positions[i]

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -self.velocity_limit, self.velocity_limit)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

            self.w = 0.4 + 0.5 * (1 - evaluations / self.budget)  # Adaptive inertia weight
            self.c1 = 1.5 + 1.5 * (evaluations / self.budget)  # Adaptive cognitive attractor
            self.c2 = 2.5 - 1.5 * (evaluations / self.budget)  # Adaptive social attractor
            
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt