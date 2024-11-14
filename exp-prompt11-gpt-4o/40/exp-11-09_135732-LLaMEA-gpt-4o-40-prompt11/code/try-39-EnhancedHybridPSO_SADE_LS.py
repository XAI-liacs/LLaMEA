import numpy as np

class EnhancedHybridPSO_SADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.9  # Adaptive inertia weight
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.mutation_factor = 0.6  # Slightly increased mutation factor
        self.crossover_rate = 0.9  # Slightly increased crossover rate

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        def adapt_inertia(evals):
            # Linearly decrease inertia weight
            return 0.4 + (0.5 * (1 - evals / self.budget))

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

            self.inertia_weight = adapt_inertia(evaluations)

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutation_factor_dynamic = self.mutation_factor + np.random.rand() * 0.2
                mutant_vector = np.clip(a + mutation_factor_dynamic * (b - c), self.lower_bound, self.upper_bound)
                trial_vector = np.copy(particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < scores[i]:
                    particles[i] = trial_vector
                    scores[i] = trial_score

            if evaluations + 1 >= self.budget:
                break

            # Enhanced Local Search Phase
            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.05, 0.05, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

                # Neighborhood search
                neighbors = particles + np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
                neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
                neighbor_scores = np.apply_along_axis(func, 1, neighbors)
                evaluations += self.population_size
                better_neighbors = neighbor_scores < scores
                particles[better_neighbors] = neighbors[better_neighbors]
                scores[better_neighbors] = neighbor_scores[better_neighbors]

        return global_best_position, global_best_score

# Example usage:
# optimizer = EnhancedHybridPSO_SADE_LS(budget=1000, dim=10)
# best_position, best_score = optimizer(some_black_box_function)