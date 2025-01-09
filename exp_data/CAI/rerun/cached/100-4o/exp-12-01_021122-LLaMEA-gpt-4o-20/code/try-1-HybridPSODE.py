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
                # Mutation with adaptive scaling factor
                indices = np.random.choice(list(range(self.population_size)), 3, replace=False)
                a, b, c = particles[indices]
                F_adaptive = self.F + (np.random.rand() - 0.5) * 0.2  # Adjust F randomly within a range
                mutant = np.clip(a + F_adaptive * (b - c), self.lower_bound, self.upper_bound)

                # Crossover and heuristic local search
                crossover_indices = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_indices, mutant, particles[i])
                heuristic_trial = trial + 0.1 * (global_best_position - trial)
                heuristic_trial = np.clip(heuristic_trial, self.lower_bound, self.upper_bound)

                # Selection with adaptive choice
                trial_score = func(trial)
                heuristic_score = func(heuristic_trial)
                evaluations += 2  # Two function evaluations

                if heuristic_score < trial_score:
                    trial_score = heuristic_score
                    trial = heuristic_trial

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