import numpy as np

class HybridPSOwithADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 10
        self.w = 0.5  # inertia
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.F = 0.5  # differential weight
        self.CR = 0.9  # crossover probability

    def initialize_particles(self):
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound,
                                           (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def update_personal_best(self, scores):
        for i in range(self.num_particles):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.positions[i]

    def update_global_best(self):
        min_index = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[min_index] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[min_index]
            self.global_best_position = self.personal_best_positions[min_index]

    def pso_step(self):
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social_component = self.c2 * r2 * (self.global_best_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

    def differential_mutation(self):
        for i in range(self.num_particles):
            indices = list(range(self.num_particles))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, mutant_vector, self.positions[i])
            trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
            yield trial_vector

    def __call__(self, func):
        self.initialize_particles()
        evaluations = 0

        while evaluations < self.budget:
            scores = np.array([func(p) for p in self.positions])
            evaluations += self.num_particles
            self.update_personal_best(scores)
            self.update_global_best()

            if evaluations >= self.budget:
                break

            self.pso_step()

            for i, trial_vector in enumerate(self.differential_mutation()):
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

                if evaluations >= self.budget:
                    break

            self.update_global_best()

        return self.global_best_position, self.global_best_score