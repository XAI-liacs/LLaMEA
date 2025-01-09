import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.F = 0.5
        self.CR = 0.9
        
    def __call__(self, func):
        np.random.seed(42)
        num_particles = self.population_size
        num_dimensions = self.dim
        budget = self.budget

        # Initialize particle positions and velocities
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (num_particles, num_dimensions))
        particle_velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
        personal_best_positions = particle_positions.copy()
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = num_particles

        while evaluations < budget:
            # Update particles
            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                particle_velocities[i] = (self.inertia_weight * particle_velocities[i] +
                                          self.cognitive_weight * r1 * (personal_best_positions[i] - particle_positions[i]) +
                                          self.social_weight * r2 * (global_best_position - particle_positions[i]))
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

                # Evaluate particle
                score = func(particle_positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particle_positions[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particle_positions[i]

            # Apply Differential Evolution
            for i in range(num_particles):
                if evaluations >= budget:
                    break
                idxs = [idx for idx in range(num_particles) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = personal_best_positions[a] + self.F * (personal_best_positions[b] - personal_best_positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(personal_best_positions[i])
                for d in range(num_dimensions):
                    if np.random.rand() < self.CR or d == np.random.randint(num_dimensions):
                        trial_vector[d] = mutant_vector[d]
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

            # Local Search (Optional, improve intensification)
            if evaluations < budget:
                local_search_idx = np.random.randint(num_particles)
                local_search_vector = global_best_position + np.random.normal(0, 0.1, num_dimensions)
                local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                local_search_score = func(local_search_vector)
                evaluations += 1
                if local_search_score < global_best_score:
                    global_best_score = local_search_score
                    global_best_position = local_search_vector

        return global_best_position