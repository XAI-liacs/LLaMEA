import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_score = np.inf
        self.iterations = self.budget // self.population_size
        self.annealing_schedule = np.linspace(1, 0, self.iterations)

    def __call__(self, func):
        eval_count = 0
        for iteration in range(self.iterations):
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    return self.global_best_position
                score = func(self.positions[i])
                eval_count += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(2)
            inertia_weight = 0.9 - 0.5 * (iteration / self.iterations)
            cognitive_component = 2.0 * r1 * (self.best_positions - self.positions)
            social_component = 2.0 * r2 * (self.global_best_position - self.positions)

            self.velocities = inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Simulated Annealing adjustment
            for i in range(self.population_size):
                if np.random.rand() < self.annealing_schedule[iteration]:
                    candidate_position = self.positions[i] + np.random.normal(0, 1, self.dim)
                    candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                    candidate_score = func(candidate_position)
                    eval_count += 1
                    if candidate_score < self.best_scores[i]:
                        self.best_scores[i] = candidate_score
                        self.best_positions[i] = candidate_position
                        if candidate_score < self.global_best_score:
                            self.global_best_score = candidate_score
                            self.global_best_position = candidate_position

        return self.global_best_position