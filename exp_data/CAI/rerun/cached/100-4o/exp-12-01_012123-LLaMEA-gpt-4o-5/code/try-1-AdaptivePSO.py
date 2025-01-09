import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 20 + 2 * int(np.sqrt(dim))
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.inertia_start = 0.9
        self.inertia_end = 0.4

    def __call__(self, func):
        np.random.seed(42)

        swarm_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        swarm_velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = np.copy(swarm_positions)
        personal_best_scores = np.full(self.num_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break

                score = func(swarm_positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm_positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm_positions[i]

            inertia_weight = self.inertia_start - (self.inertia_start - self.inertia_end) * (evaluations / self.budget)

            for i in range(self.num_particles):
                r1 = np.random.uniform(size=self.dim)
                r2 = np.random.uniform(size=self.dim)

                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - swarm_positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - swarm_positions[i])

                swarm_velocities[i] = inertia_weight * swarm_velocities[i] + cognitive_velocity + social_velocity
                swarm_positions[i] += swarm_velocities[i]

                # Enforce boundaries
                swarm_positions[i] = np.clip(swarm_positions[i], self.lower_bound, self.upper_bound)

            # Local restart for exploration
            if evaluations < self.budget and evaluations % (self.budget // 5) == 0:
                local_restart_index = np.random.choice(self.num_particles)
                swarm_positions[local_restart_index] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        return global_best_position, global_best_score