import numpy as np

class QIPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.positions])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def update_particles(self, lb, ub):
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(2)
            cognitive_term = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_term = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i]) + cognitive_term + social_term
            
            # Quantum inspired position update
            self.positions[i] = self.positions[i] + self.velocities[i] + \
                                np.sin(self.velocities[i]) * (self.global_best_position - self.positions[i])
            self.positions[i] = np.clip(self.positions[i], lb, ub)

            current_score = self.evaluate(self.positions[i])
            if current_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = current_score
                self.personal_best_positions[i] = self.positions[i].copy()
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_swarm(lb, ub)

        while self.evaluations < self.budget:
            self.update_particles(lb, ub)
            self.evaluations += self.swarm_size
            if self.evaluations >= self.budget:
                break

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}