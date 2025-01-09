import numpy as np

class ChaoticAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0
        self.adaptation_rate = 0.1
        self.chaotic_sequence = self.generate_chaotic_sequence(self.budget, logistic_map=True)

    def generate_chaotic_sequence(self, length, logistic_map=False):
        x = np.random.rand()
        sequence = []
        for _ in range(length):
            if logistic_map:
                x = 4 * x * (1 - x)  # Logistic map
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        self.initialize_particles(bounds, func)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                r1 = self.chaotic_sequence[self.evaluations % len(self.chaotic_sequence)]
                r2 = np.random.rand(self.dim)

                self.c1 = 1.2 + (0.3 * r1)  # Dynamic adjustment with chaotic influence
                self.c2 = 1.2 + (0.3 * np.random.rand())

                scaling_factor = np.random.uniform(0.5, 1.5, size=self.dim)
                self.velocities[i] = (self.w * self.velocities[i] * scaling_factor +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]))

                perturbation = np.random.randn(self.dim) * 0.01 * np.exp(-self.evaluations/self.budget) * (bounds[:, 1] - bounds[:, 0])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] + perturbation, bounds[:, 0], bounds[:, 1])

                score = self.evaluate(func, self.particles[i])
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()

                opposite_particle = bounds[:, 1] + bounds[:, 0] - self.particles[i]
                opposite_score = self.evaluate(func, opposite_particle)
                if opposite_score < score:
                    self.particles[i] = opposite_particle
                    if opposite_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = opposite_score
                        self.personal_best_positions[i] = opposite_particle.copy()
                    if opposite_score < self.global_best_score:
                        self.global_best_score = opposite_score
                        self.global_best_position = opposite_particle.copy()

                diversity = np.mean(np.std(self.particles, axis=0))
                self.w = max(0.4, 0.9 - (0.9 - 0.4) * (self.evaluations/self.budget) * (1 + 0.5 * diversity))

        return self.global_best_position

    def initialize_particles(self, bounds, func):
        self.particles = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        self.velocities = np.random.randn(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) * 0.1
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.array([self.evaluate(func, p) for p in self.particles])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            return float('inf')
        self.evaluations += 1
        return func(individual)