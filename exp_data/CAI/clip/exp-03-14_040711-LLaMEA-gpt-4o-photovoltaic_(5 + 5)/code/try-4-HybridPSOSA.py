import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 10)
        self.position = np.random.rand(self.num_particles, dim)
        self.velocity = np.zeros((self.num_particles, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.temperature = 1.0

    def update_velocity(self, inertia=0.5, cognitive=1.5, social=1.5):
        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        cognitive_velocity = cognitive * r1 * (self.personal_best_position - self.position)
        social_velocity = social * r2 * (self.global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds.lb, bounds.ub)

    def anneal(self):
        self.temperature *= 0.99

    def acceptance_probability(self, candidate_value):
        if candidate_value < self.global_best_value:
            return 1.0
        else:
            return np.exp((self.global_best_value - candidate_value) / self.temperature)

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                candidate_value = func(self.position[i])
                evaluations += 1

                if candidate_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = candidate_value
                    self.personal_best_position[i] = self.position[i]

                if self.global_best_position is None or candidate_value < self.global_best_value:
                    self.global_best_value = candidate_value
                    self.global_best_position = self.position[i]

            self.update_velocity()
            self.update_position(bounds)
            self.anneal()

            for i in range(self.num_particles):
                candidate_value = func(self.position[i])
                evaluations += 1

                if np.random.rand() < self.acceptance_probability(candidate_value):
                    self.position[i] = self.position[i]

        return self.global_best_position