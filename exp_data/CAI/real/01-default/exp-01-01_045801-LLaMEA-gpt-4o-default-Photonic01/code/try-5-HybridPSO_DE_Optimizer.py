import numpy as np

class HybridPSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.F = 0.5  # DE mutation factor
        self.CR = 0.7  # DE crossover probability
        self.velocities = None
        self.positions = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')

    def initialize_particles(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([self.evaluate(p) for p in self.positions])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def update_particles(self, lb, ub):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                                  self.c2 * r2 * (self.global_best_position - self.positions[i]))
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)
            score = self.evaluate(self.positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

    def differential_evolution(self, lb, ub):
        for i in range(self.num_particles):
            indices = np.random.choice(self.num_particles, 3, replace=False)
            x1, x2, x3 = self.positions[indices]
            trial = x1 + self.F * (x2 - x3)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, trial, self.positions[i])
            trial = np.clip(trial, lb, ub)
            score = self.evaluate(trial)
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = trial.copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = trial.copy()

    def update_inertia_weight(self, evaluations):
        self.inertia_weight = (self.inertia_weight_max - 
                               (self.inertia_weight_max - self.inertia_weight_min) 
                               * (evaluations / self.budget))

    def __call__(self, func):
        self.func = func
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        evaluations = 0

        self.inertia_weight_max = self.inertia_weight
        self.initialize_particles(self.lb, self.ub)

        while evaluations < self.budget:
            self.update_particles(self.lb, self.ub)
            self.differential_evolution(self.lb, self.ub)
            evaluations += self.num_particles
            self.update_inertia_weight(evaluations)

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}