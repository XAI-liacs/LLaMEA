import numpy as np

class AdaptiveLearningPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.base_inertia_weight = 0.9
        self.cognitive_base = 2.0
        self.social_base = 2.0
        self.learning_rates = np.full(self.population_size, 0.5)
        self.stagnation_count = np.zeros(self.population_size)
        self.adaptive_threshold = 0.01

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(np.clip(self.particles[i], lb, ub))
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                    self.stagnation_count[i] = 0  # Reset stagnation for improved particles
                else:
                    self.stagnation_count[i] += 1

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            if eval_count >= self.budget:
                break

            for i in range(self.population_size):
                adaptive_inertia = self.base_inertia_weight - (self.base_inertia_weight - 0.4) * (eval_count / self.budget)
                cognitive_component = self.learning_rates[i] * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.learning_rates[i] * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
                self.velocities[i] = adaptive_inertia * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]

                if self.stagnation_count[i] > 5:  # If stagnated, adjust learning rate
                    self.learning_rates[i] = min(1.0, self.learning_rates[i] + self.adaptive_threshold)
                else:
                    self.learning_rates[i] = max(0.1, self.learning_rates[i] - self.adaptive_threshold)
            
        return self.global_best_position, self.global_best_score