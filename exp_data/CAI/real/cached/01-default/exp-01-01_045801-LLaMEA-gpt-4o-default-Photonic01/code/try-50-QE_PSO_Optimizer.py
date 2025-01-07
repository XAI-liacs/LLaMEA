import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.c1 = 1.49618  # Cognitive coefficient
        self.c2 = 1.49618  # Social coefficient
        self.w_max = 0.9   # Max inertia weight
        self.w_min = 0.4   # Min inertia weight
        self.particles = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_swarm(self, lb, ub):
        self.particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.array([self.evaluate(ind) for ind in self.particles])
        self.update_global_best()

    def evaluate(self, particle):
        return self.func(particle)

    def update_global_best(self):
        best_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[best_idx] < self.global_best_score:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_score = self.personal_best_scores[best_idx]

    def quantum_enhancement(self):
        quantum_positions = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        for i in range(self.swarm_size):
            quantum_superposition = np.random.rand(self.dim)
            self.particles[i] = self.global_best + quantum_superposition * (self.particles[i] - self.global_best)
            self.particles[i] = np.clip(self.particles[i], self.func.bounds.lb, self.func.bounds.ub)

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_swarm(lb, ub)

        while self.evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best - self.particles[i])

                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component
                self.particles[i] = self.particles[i] + self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

                current_score = self.evaluate(self.particles[i])
                self.evaluations += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best[i] = self.particles[i].copy()
                    self.personal_best_scores[i] = current_score

            self.update_global_best()

            if self.evaluations + self.swarm_size < self.budget:
                self.quantum_enhancement()
                self.evaluations += self.swarm_size

        return {'solution': self.global_best, 'fitness': self.global_best_score}