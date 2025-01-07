import numpy as np

class HybridSwarmDifferential:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(10 * dim, 100)  # Adaptive population size
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.diff_weight = 0.8
        self.crossover_prob = 0.7  # Initial crossover probability
        self.evaluations = 0

    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.particles = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')

    def update_personal_bests(self, func):
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            score = func(self.particles[i])
            self.evaluations += 1
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()

    def pso_update(self):
        diversity = np.var(self.particles, axis=0).mean()
        dynamic_inertia = self.inertia_weight - 0.4 * (diversity / (diversity + 1e-10))
        self.social_coeff = 1.5 + 0.5 * (diversity / (diversity + 1e-10)) # Dynamically adjust social coefficient
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_velocity = self.social_coeff * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = (dynamic_inertia * self.velocities[i] +
                                  cognitive_velocity + social_velocity)
            self.particles[i] += self.velocities[i]

    def differential_evolution(self, bounds, func):
        self.crossover_prob = 0.9 if self.evaluations > self.budget // 3 else 0.5 # More adaptive crossover
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.diff_weight * (b - c), bounds.lb, bounds.ub)
            trial_vector = np.copy(self.particles[i])
            crossover_mask = np.random.rand(self.dim) < self.crossover_prob
            trial_vector[crossover_mask] = mutant_vector[crossover_mask]
            if func(trial_vector) < func(self.particles[i]):
                self.particles[i] = trial_vector
            self.evaluations += 1

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_particles(bounds)
        while self.evaluations < self.budget:
            self.update_personal_bests(func)
            self.pso_update()
            self.differential_evolution(bounds, func)
        return self.global_best_position, self.global_best_score