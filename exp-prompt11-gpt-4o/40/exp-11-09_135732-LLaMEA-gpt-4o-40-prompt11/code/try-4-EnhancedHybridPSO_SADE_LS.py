import numpy as np

class EnhancedHybridPSO_SADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.9  # Adaptive inertia weight range
        self.inertia_min = 0.4
        self.cognitive_constant = 2.0  # Enhanced cognitive and social constants
        self.social_constant = 2.0
        self.mutation_factor = 0.6
        self.crossover_rate = 0.9  # Slightly higher crossover rate
        self.chaos_coefficient = 0.7

    def chaotic_map(self, x):
        return (4.0 * x) * (1.0 - x)

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        inertia_weight = self.inertia_weight

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles)
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            inertia_weight = max(self.inertia_min, inertia_weight * 0.99)  # Adaptive inertia weight reduction
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                chaos_seed = np.random.rand()
                chaos_value = self.chaotic_map(chaos_seed)
                mutant_vector = np.clip(a + self.mutation_factor * (b - c) + chaos_value * self.chaos_coefficient, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < scores[i]:
                    particles[i] = trial_vector
                    scores[i] = trial_score

            if evaluations + 2 >= self.budget:
                break

            # Enhanced Local Search Phase with dynamic step size
            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                step_size = np.random.uniform(0.05, 0.15)
                local_candidate = particles[i] + step_size * np.random.uniform(-1, 1, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

        return global_best_position, global_best_score