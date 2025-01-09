import numpy as np

class EnhancedPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.f_opt = np.Inf
        self.x_opt = None
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.inertia_max = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)
        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                f_val = func(positions[i])
                evaluations += 1

                if f_val < personal_best_scores[i]:
                    personal_best_scores[i] = f_val
                    personal_best_positions[i] = positions[i]

                if f_val < global_best_score:
                    global_best_score = f_val
                    global_best_position = positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            self.inertia_weight = (self.inertia_max - self.inertia_min) * (self.budget - evaluations) / self.budget + self.inertia_min
            velocities = (self.inertia_weight * velocities
                          + self.c1 * r1 * (personal_best_positions - positions)
                          + self.c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            # Local search phase
            if evaluations + self.swarm_size <= self.budget:
                for i in range(self.swarm_size):
                    local_search_step = np.random.uniform(-0.1, 0.1, self.dim)
                    candidate_position = np.clip(positions[i] + local_search_step, lb, ub)
                    candidate_score = func(candidate_position)
                    evaluations += 1

                    if candidate_score < personal_best_scores[i]:
                        personal_best_scores[i] = candidate_score
                        personal_best_positions[i] = candidate_position

                    if candidate_score < global_best_score:
                        global_best_score = candidate_score
                        global_best_position = candidate_position

            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt