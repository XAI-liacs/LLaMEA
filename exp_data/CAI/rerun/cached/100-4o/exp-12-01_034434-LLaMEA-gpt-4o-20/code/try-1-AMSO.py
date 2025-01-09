import numpy as np

class AMSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_swarms = 3  # Number of swarms
        self.swarm_size = 10  # Number of particles in each swarm
        self.num_particles = self.num_swarms * self.swarm_size
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.5   # Inertia weight
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_score = np.inf
        self.current_evaluations = 0

    def __call__(self, func):
        while self.current_evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(self.positions[i])
                self.current_evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            self.adaptive_control()

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.global_best_position, self.global_best_score

    def adaptive_control(self):
        # Dynamically adjust inertia weight and learning factors based on performance
        if self.current_evaluations % (self.budget // 10) == 0:
            progress = self.global_best_score / np.min(self.personal_best_scores)
            self.w = 0.9 - progress * 0.4
            self.c1 = 2.5 - progress * 0.5
            self.c2 = 1.5 + progress * 0.5