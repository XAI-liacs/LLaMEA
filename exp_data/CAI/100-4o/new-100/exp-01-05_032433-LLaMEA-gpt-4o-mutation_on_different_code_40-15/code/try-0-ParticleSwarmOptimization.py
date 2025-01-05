import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, omega=0.5, phi_p=0.5, phi_g=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particle positions and velocities
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)
        global_best_position = None
        global_best_score = np.Inf

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                score = func(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, self.dim)
                r_g = np.random.uniform(0, 1, self.dim)
                velocities[i] = (self.omega * velocities[i] +
                                 self.phi_p * r_p * (personal_best_positions[i] - positions[i]) +
                                 self.phi_g * r_g * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt