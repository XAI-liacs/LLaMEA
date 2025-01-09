import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.9
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.particles])
        self.evaluations += self.num_particles
        for i in range(self.num_particles):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.particles[i]
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.particles[i]
    
    def update_particles(self):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(2)
            self.velocities[i] = (self.w * self.velocities[i] 
                                  + self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) 
                                  + self.c2 * r2 * (self.global_best_position - self.particles[i]))
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
        if self.evaluations > self.budget // 2:  # Dynamic resizing
            self.num_particles = max(20, self.num_particles - 5)

    def differential_evolution(self, i):
        indices = [idx for idx in range(self.num_particles) if idx != i]
        a, b, c = np.random.choice(indices, 3, replace=False)
        self.F = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation scaling
        mutant_vector = self.particles[a] + self.F * (self.particles[b] - self.particles[c])
        trial_vector = np.copy(self.particles[i])
        for j in range(self.dim):
            if np.random.rand() < self.CR:
                trial_vector[j] = mutant_vector[j]
        return np.clip(trial_vector, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.evaluate(func)
            if self.evaluations >= self.budget:
                break
            self.update_particles()
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.differential_evolution(i)
                trial_score = func(trial_vector)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector
        return self.global_best_position, self.global_best_score