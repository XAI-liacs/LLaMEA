import numpy as np

class HybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, self.budget)  # Number of particles in the swarm
        self.particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
        self.personal_bests = self.particles.copy()
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        inertia_weight = 0.7
        cognitive_weight = 1.5
        social_weight = 1.5

        while self.evaluations < self.budget:
            # Evaluate all particles
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break
                score = func(self.particles[i])
                self.evaluations += 1

                # Update personal bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_bests[i] = self.particles[i].copy()

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            # Update velocities and positions
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_component = cognitive_weight * r1 * (self.personal_bests[i] - self.particles[i])
                social_component = social_weight * r2 * (self.global_best - self.particles[i])

                self.velocities[i] = (
                    inertia_weight * self.velocities[i] + cognitive_component + social_component
                )

                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)

            # Apply adaptive differential evolution
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break
                # Select three random indices different from i
                indices = np.random.choice(list(set(range(self.num_particles)) - {i}), 3, replace=False)
                a, b, c = self.particles[indices]
                F = 0.5 + np.random.rand() * 0.5  # Differential weight

                mutant = a + F * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)

                crossover_rate = 0.9
                crossover = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover, mutant, self.particles[i])

                score = func(trial)
                self.evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_bests[i] = trial.copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = trial.copy()

        return self.global_best, self.global_best_score