import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_coeff = 0.5

    def initialize_population(self, lb, ub):
        self.particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.array([self.evaluate(p) for p in self.particles])
        min_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[min_idx].copy()
        self.global_best_score = self.personal_best_scores[min_idx]

    def evaluate(self, particle):
        score = self.func(particle)
        self.evaluations += 1
        return score

    def update_particles(self, lb, ub):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_velocity = self.social_coeff * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_velocity + social_velocity)
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], lb, ub)

            current_score = self.evaluate(self.particles[i])
            if current_score < self.personal_best_scores[i]:
                self.personal_best_positions[i] = self.particles[i].copy()
                self.personal_best_scores[i] = current_score
                if current_score < self.global_best_score:
                    self.global_best_position = self.particles[i].copy()
                    self.global_best_score = current_score

    def quantum_entanglement_adjustment(self, lb, ub):
        entangled_positions = np.array([
            self.global_best_position + self.quantum_coeff * np.random.rand(self.dim) *
            (self.personal_best_positions[i] - self.global_best_position)
            for i in range(self.population_size)
        ])
        entangled_positions = np.clip(entangled_positions, lb, ub)
        for i in range(self.population_size):
            score = self.evaluate(entangled_positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_positions[i] = entangled_positions[i].copy()
                self.personal_best_scores[i] = score
                if score < self.global_best_score:
                    self.global_best_position = entangled_positions[i].copy()
                    self.global_best_score = score

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.update_particles(lb, ub)
            self.quantum_entanglement_adjustment(lb, ub)

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}