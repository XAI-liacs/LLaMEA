import numpy as np

class AdaptiveHybridSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 40
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.adaptive_rate = 0.1
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.population_size

            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            if evals >= self.budget:
                break

            adaptive_local_search = np.random.random() < self.adaptive_rate
            if adaptive_local_search:
                perturbation = np.random.normal(0, 0.1, self.dim)
                candidate_position = self.global_best_position + perturbation
                candidate_position = np.clip(candidate_position, self.bounds[0], self.bounds[1])
                candidate_score = func(candidate_position)
                evals += 1
                if candidate_score < self.global_best_score:
                    self.global_best_score = candidate_score
                    self.global_best_position = candidate_position

            r1, r2 = np.random.rand(2)
            self.velocities = (self.inertia_weight * self.velocities +
                               self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions) +
                               self.social_coeff * r2 * (self.global_best_position - self.positions))
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        return self.global_best_position