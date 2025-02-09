import numpy as np

class SelfAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w = 0.7   # Inertia weight
        self.population = []
        self.velocities = []

    def initialize_population(self, lb, ub):
        for _ in range(self.population_size):
            position = np.random.uniform(lb, ub, self.dim)
            velocity = np.random.uniform(-1, 1, self.dim)
            score = float('inf')
            self.population.append((position, score, position))  # (position, score, best_position)
            self.velocities.append(velocity)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        global_best_position = None
        global_best_score = float('inf')

        while evaluations < self.budget:
            for i in range(self.population_size):
                current_position, current_score, personal_best_position = self.population[i]

                # Evaluate current position
                score = func(current_position)
                evaluations += 1

                if score < current_score:
                    self.population[i] = (current_position, score, current_position)

                    if score < global_best_score:
                        global_best_position = current_position
                        global_best_score = score

                # Update particle velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                personal_best_position = self.population[i][2]
                inertia = self.w * self.velocities[i]
                cognitive = self.c1 * r1 * (personal_best_position - current_position)
                social = self.c2 * r2 * (global_best_position - current_position)
                velocity = inertia + cognitive + social

                # Update position and clip to bounds
                new_position = current_position + velocity
                new_position = np.clip(new_position, lb, ub)

                self.velocities[i] = velocity
                self.population[i] = (new_position, score, personal_best_position)

            # Adapt inertia weight and coefficients based on diversity
            positions = np.array([ind[0] for ind in self.population])
            diversity = np.std(positions, axis=0).mean()

            self.w = 0.5 + 0.2 * np.exp(-diversity)
            self.c1 = 2.5 - 0.5 * diversity
            self.c2 = 2.5 - 0.5 * (1 - diversity)

        return global_best_position