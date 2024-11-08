import numpy as np

class OptimizedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.inertia_weight = 0.5
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([float('inf')] * self.population_size)

        for i in range(self.population_size):
            score = func(particles[i])
            personal_best_scores[i] = score

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        eval_count = self.population_size

        while eval_count < self.budget:
            r1, r2 = np.random.rand(self.population_size), np.random.rand(self.population_size)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coef * r1[:, None] * (personal_best_positions - particles)
                          + self.social_coef * r2[:, None] * (global_best_position - particles))
                          
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = np.random.permutation(self.population_size)
                idxs = idxs[idxs != i][:3]
                a, b, c = particles[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)

                jrand = np.random.randint(self.dim)
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                crossover_mask[jrand] = True
                trial_vector = np.where(crossover_mask, mutant_vector, particles[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < personal_best_scores[global_best_index]:
                    global_best_position = trial_vector
                    global_best_index = i

        return global_best_position