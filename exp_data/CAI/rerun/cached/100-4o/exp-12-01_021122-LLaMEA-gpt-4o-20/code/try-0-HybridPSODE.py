import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, self.budget // 2)
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        self.w = 0.7   # inertia weight
        self.F = 0.5   # differential weight
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            # PSO phase
            r1, r2 = np.random.rand(2)
            velocities = self.w * velocities + self.c1 * r1 * (personal_best_positions - particles) + self.c2 * r2 * (global_best_position - particles)
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            # DE phase
            new_population = np.copy(particles)
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(list(range(self.population_size)), 3, replace=False)
                a, b, c = particles[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_indices = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_indices, mutant, particles[i])

                # Selection
                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

            if evaluations >= self.budget:
                break

            particles = np.copy(personal_best_positions)

        return global_best_position