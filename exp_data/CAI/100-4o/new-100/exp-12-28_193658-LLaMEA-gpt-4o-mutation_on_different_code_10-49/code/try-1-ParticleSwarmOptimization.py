import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.9, cognitive=1.5, social=1.5):
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
        personal_best_positions = positions.copy()
        personal_best_values = np.full(self.swarm_size, np.Inf)
        global_best_position = None
        global_best_value = np.Inf

        evals = 0
        while evals < self.budget:
            for i in range(self.swarm_size):
                f_value = func(positions[i])
                evals += 1
                
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i].copy()

                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = positions[i].copy()

                if evals >= self.budget:
                    break

            w = self.inertia * (0.4 + 0.5 * (1 - evals / self.budget))  # Line changed
            c1 = self.cognitive
            c2 = self.social
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)  # Line changed

            for i in range(self.swarm_size):
                velocities[i] = (w * velocities[i] +
                                 c1 * r1[i] * (personal_best_positions[i] - positions[i]) +
                                 c2 * r2[i] * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

        self.f_opt, self.x_opt = global_best_value, global_best_position
        return self.f_opt, self.x_opt