import numpy as np

class HybridADE_SI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 4 + int(3 * np.log(self.dim)) * self.dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Initial crossover probability
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate the current population
            scores = np.apply_along_axis(func, 1, self.particles)
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.particles[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.particles[i]

            if self.evaluations >= self.budget:
                break

            # Update crossover probability adaptively
            self.cr = 0.8 + 0.2 * (self.global_best_score / (self.global_best_score + 1))

            # Differential Evolution with Swarm Intelligence
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.particles[indices]
                mutant = a + self.f * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Swarm Intelligence inspired velocity update
                inertia = 0.7 * self.velocities[i]
                cognitive_component = 0.1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
                social_component = 0.1 * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
                self.velocities[i] = inertia + cognitive_component + social_component

                # Crossover
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.particles[i] + self.velocities[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < scores[i]:
                    self.particles[i] = trial
                    scores[i] = trial_score

        return self.global_best_position, self.global_best_score