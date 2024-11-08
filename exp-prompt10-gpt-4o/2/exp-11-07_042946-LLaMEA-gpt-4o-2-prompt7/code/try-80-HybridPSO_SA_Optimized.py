import numpy as np

class HybridPSO_SA_Optimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf
        self.alpha = 0.8
        self.beta = 0.15
        self.temperature = 100.0

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.population_size):
                current_score = func(self.positions[i])
                eval_count += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]

            if eval_count >= self.budget:
                break

            # Update velocities and positions
            r = np.random.rand(self.population_size, self.dim)
            self.velocities = 0.45 * self.velocities + r * (self.alpha * (self.personal_best_positions - self.positions) + self.beta * (self.global_best_position - self.positions))
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)

            # Apply Simulated Annealing to refine global best
            proposed_position = self.global_best_position + np.random.uniform(-1, 1, self.dim)
            proposed_position = np.clip(proposed_position, -5.0, 5.0)
            proposed_score = func(proposed_position)
            eval_count += 1

            acceptance_probability = np.exp((self.global_best_score - proposed_score) / self.temperature)
            if proposed_score < self.global_best_score or acceptance_probability > np.random.rand():
                self.global_best_position = proposed_position
                self.global_best_score = proposed_score

            self.temperature *= 0.96  # refined cooling schedule

        return self.global_best_position, self.global_best_score