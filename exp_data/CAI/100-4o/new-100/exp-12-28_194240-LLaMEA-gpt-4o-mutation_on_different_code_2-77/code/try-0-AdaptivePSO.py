import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.swarm_size, np.Inf)
        global_best_position = None

        def evaluate_particle(particle):
            return func(particle)

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                score = evaluate_particle(swarm[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = swarm[i]
            
            if global_best_position is None or self.f_opt < evaluate_particle(global_best_position):
                global_best_position = np.copy(self.x_opt)

            w = 0.5 + np.random.rand() / 2  # Dynamic inertia weight
            c1 = 1.5
            c2 = 1.5
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 c2 * r2 * (global_best_position - swarm[i]))
                swarm[i] += velocities[i]

                # Mutation operator
                if np.random.rand() < 0.1:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    swarm[i] += mutation_vector

                swarm[i] = np.clip(swarm[i], lb, ub)
        
        return self.f_opt, self.x_opt