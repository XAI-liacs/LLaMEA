import numpy as np

class MultiStrategyDynamicSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.strategies = [
            {'name': 'default', 'omega': 0.5, 'phi_p': 1.5, 'phi_g': 1.5},
            {'name': 'aggressive', 'omega': 0.7, 'phi_p': 2.0, 'phi_g': 2.0},
            {'name': 'conservative', 'omega': 0.3, 'phi_p': 1.2, 'phi_g': 1.2},
        ]
        self.strategy_scores = np.zeros(len(self.strategies))
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            # Select strategy based on performance feedback
            prob_distribution = self.strategy_scores + 1  # Avoid zero probability
            prob_distribution /= prob_distribution.sum()
            strategy_idx = np.random.choice(len(self.strategies), p=prob_distribution)
            strategy = self.strategies[strategy_idx]

            cognitive_component = strategy['phi_p'] * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social_component = strategy['phi_g'] * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            inertia = strategy['omega']

            self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], lb, ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.particles[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                    # Update strategy score positively
                    self.strategy_scores[strategy_idx] += 1

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score