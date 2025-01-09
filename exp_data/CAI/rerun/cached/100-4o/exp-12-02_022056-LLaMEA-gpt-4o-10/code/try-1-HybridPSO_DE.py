import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 10 + 2 * self.dim
        self.particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.f = 0.5  # Differential Evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.iteration = 0

    def evaluate(self, func):
        for i in range(self.pop_size):
            score = func(self.particles[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i]

    def update_particles(self):
        inertia_weight = 0.7
        cognitive_constant = 1.5
        social_constant = 1.5
        r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
        
        # Velocity update
        self.velocities = (inertia_weight * self.velocities +
                           cognitive_constant * r1 * (self.personal_best_positions - self.particles) +
                           social_constant * r2 * (self.global_best_position - self.particles))
        
        # Position update
        self.particles += self.velocities
        self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])

    def differential_evolution(self, func):  # Added func parameter
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])
            crossover = np.random.rand(self.dim) < self.cr
            trial = np.where(crossover, mutant, self.particles[i])

            trial_score = func(trial)  # Corrected function call
            if trial_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = trial_score
                self.personal_best_positions[i] = trial
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate(func)
            self.update_particles()
            self.differential_evolution(func)  # Corrected function call
            evaluations += self.pop_size
        return self.global_best_position, self.global_best_score