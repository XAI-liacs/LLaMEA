import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.5  # Scaling factor for DE
        self.cr = 0.9  # Crossover probability for DE
        self.inertia_weight = 0.7  # Inertia weight for PSO
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                candidates = list(range(i)) + list(range(i + 1, self.population_size))
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant_vector = np.clip(population[a] + self.f * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Evaluate trial vector
                trial_score = func(trial_vector)
                eval_count += 1

                # Selection
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                    # Update global best
                    if trial_score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = trial_vector

                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_component = self.inertia_weight * velocities[i]
                cognitive_component = self.cognitive_constant * r1 * (personal_best_positions[i] - population[i])
                social_component = self.social_constant * r2 * (global_best_position - population[i])
                velocities[i] = inertia_component + cognitive_component + social_component
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                if eval_count >= self.budget:
                    break

        return global_best_position