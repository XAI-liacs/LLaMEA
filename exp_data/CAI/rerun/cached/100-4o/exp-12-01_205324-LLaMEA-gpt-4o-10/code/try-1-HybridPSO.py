import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        np.random.seed()
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_idx]
        global_best_score = personal_best_scores[best_idx]

        evaluations = self.population_size

        def mutate(target_idx):
            indexes = [idx for idx in range(self.population_size) if idx != target_idx]
            a, b, c = np.random.choice(indexes, 3, replace=False)
            mutant_vector = particles[a] + self.mutation_factor * (particles[b] - particles[c])
            return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity and position using PSO rules
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.social_weight * r2 * (global_best_position - particles[i])
                velocities[i] = (self.inertia * velocities[i] +
                                 cognitive_component +
                                 social_component)
                particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Differential mutation
                mutant_vector = mutate(i)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, particles[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                # Evaluate the trial vector
                trial_score = func(trial_vector)
                evaluations += 1

                # Replace if the trial vector is better
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                    # Update global best if necessary
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score