import numpy as np

class HybridPSO_SADE_LS_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.dynamic_population = True

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles)
            evaluations += population_size

            for i in range(population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            inertia_weight_dynamic = 0.4 + 0.5 * (1 - evaluations / self.budget)
            r1 = np.random.rand(population_size, self.dim)
            r2 = np.random.rand(population_size, self.dim)
            velocities = (inertia_weight_dynamic * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutation_factor_dynamic = self.mutation_factor + np.random.rand() * 0.3
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

            # Local Search Phase
            for i in range(population_size):
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.1, 0.1, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

            if self.dynamic_population and evaluations < self.budget / 2:
                population_size = max(10, int(self.initial_population_size * (1 - evaluations / self.budget)))

        return global_best_position, global_best_score

# Example usage:
# optimizer = HybridPSO_SADE_LS_Enhanced(budget=1000, dim=10)
# best_position, best_score = optimizer(some_black_box_function)