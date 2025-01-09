import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.global_best = None
        self.global_best_score = float('inf')

    def __call__(self, func):
        # Initialize the particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        func_evals = 0

        def evaluate(particle):
            nonlocal func_evals
            if func_evals < self.budget:
                func_evals += 1
                return func(particle)
            return float('inf')

        while func_evals < self.budget:
            # Update personal bests and global best
            for i in range(self.pop_size):
                score = evaluate(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = particles[i]

            # PSO update
            inertia = 0.5 + np.random.rand() / 2
            cognitive_coeff = 1.5
            social_coeff = 1.5
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    inertia * velocities[i] +
                    cognitive_coeff * r1 * (personal_best_positions[i] - particles[i]) +
                    social_coeff * r2 * (self.global_best - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            # DE mutation and crossover
            for i in range(self.pop_size):
                if func_evals >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, particles[i])
                trial_score = evaluate(trial)
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

        return self.global_best