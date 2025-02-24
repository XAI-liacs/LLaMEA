import numpy as np

class AdaptiveMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = 5 * dim
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.inertia = 0.7
        self.velocities = None
        self.positions = None
        self.best_personal_positions = None
        self.best_personal_scores = None
        self.best_swarm_position = np.random.uniform(-1, 1, dim)
        self.best_swarm_score = float('-inf')

    def initialize_swarms(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.num_swarms, self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_swarms, self.swarm_size, self.dim))
        self.best_personal_positions = np.copy(self.positions)
        self.best_personal_scores = np.full((self.num_swarms, self.swarm_size), float('-inf'))

    def update_particles(self, swarm_idx, func, global_best_pos):
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (self.best_personal_positions[swarm_idx, i] - self.positions[swarm_idx, i])
            social = self.c2 * r2 * (global_best_pos - self.positions[swarm_idx, i])
            self.velocities[swarm_idx, i] = self.inertia * self.velocities[swarm_idx, i] + cognitive + social
            self.positions[swarm_idx, i] += self.velocities[swarm_idx, i]
            
            # Enforce boundaries
            self.positions[swarm_idx, i] = np.clip(self.positions[swarm_idx, i], func.bounds.lb, func.bounds.ub)

            # Calculate fitness
            fitness = func(self.positions[swarm_idx, i]) - self.periodic_cost_function(self.positions[swarm_idx, i])

            # Update personal best
            if fitness > self.best_personal_scores[swarm_idx, i]:
                self.best_personal_scores[swarm_idx, i] = fitness
                self.best_personal_positions[swarm_idx, i] = self.positions[swarm_idx, i]

            # Update global best
            if fitness > self.best_swarm_score:
                self.best_swarm_score = fitness
                self.best_swarm_position = self.positions[swarm_idx, i]

    def periodic_cost_function(self, candidate):
        periodic_penalty = np.var(np.diff(candidate.reshape(-1, 2), axis=0))
        return periodic_penalty

    def optimize(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for s in range(self.num_swarms):
                self.update_particles(s, func, self.best_swarm_position)
                evaluations += self.swarm_size
                if evaluations >= self.budget:
                    break
        return self.best_swarm_position

    def __call__(self, func):
        self.initialize_swarms(func.bounds.lb, func.bounds.ub)
        return self.optimize(func)