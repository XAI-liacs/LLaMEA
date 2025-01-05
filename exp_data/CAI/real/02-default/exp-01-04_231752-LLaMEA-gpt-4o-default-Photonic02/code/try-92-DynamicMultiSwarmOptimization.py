import numpy as np

class DynamicMultiSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = 10 + self.dim
        self.swarms = [[] for _ in range(self.num_swarms)]
        self.velocities = [[] for _ in range(self.num_swarms)]
        self.personal_best_positions = [[] for _ in range(self.num_swarms)]
        self.personal_best_scores = [[] for _ in range(self.num_swarms)]
        self.global_best_position = None
        self.global_best_score = np.inf
        self.omega = 0.5
        self.phi_p = 1.5
        self.phi_g = 1.5
        self.interaction_strengths = np.linspace(0.1, 0.5, self.num_swarms)

    def _initialize_swarms(self, lb, ub):
        for swarm in range(self.num_swarms):
            self.swarms[swarm] = np.random.rand(self.swarm_size, self.dim) * (ub - lb) + lb
            self.velocities[swarm] = np.random.rand(self.swarm_size, self.dim) * (ub - lb) / 10
            self.personal_best_positions[swarm] = np.copy(self.swarms[swarm])
            self.personal_best_scores[swarm] = np.full(self.swarm_size, np.inf)

    def _update_particles(self, lb, ub):
        for swarm in range(self.num_swarms):
            for i in range(self.swarm_size):
                cognitive_component = self.phi_p * np.random.rand(self.dim) * (self.personal_best_positions[swarm][i] - self.swarms[swarm][i])
                social_component = self.phi_g * np.random.rand(self.dim) * (self.global_best_position - self.swarms[swarm][i])
                inertia = self.omega

                interaction_component = np.zeros(self.dim)
                for other_swarm in range(self.num_swarms):
                    if other_swarm != swarm:
                        interaction_component += self.interaction_strengths[swarm] * np.random.rand(self.dim) * (np.mean(self.swarms[other_swarm], axis=0) - self.swarms[swarm][i])
                
                self.velocities[swarm][i] = inertia * self.velocities[swarm][i] + cognitive_component + social_component + interaction_component
                self.swarms[swarm][i] += self.velocities[swarm][i]
                self.swarms[swarm][i] = np.clip(self.swarms[swarm][i], lb, ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_swarms(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.swarm_size):
                    if eval_count >= self.budget:
                        break

                    score = func(self.swarms[swarm][i])
                    eval_count += 1

                    if score < self.personal_best_scores[swarm][i]:
                        self.personal_best_scores[swarm][i] = score
                        self.personal_best_positions[swarm][i] = self.swarms[swarm][i]

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.swarms[swarm][i]

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score