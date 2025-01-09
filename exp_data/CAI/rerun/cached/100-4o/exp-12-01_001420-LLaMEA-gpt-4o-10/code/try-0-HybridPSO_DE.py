import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))  # Adaptive population size
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.w = 0.5   # Inertia weight
        self.CR = 0.9  # Crossover rate for DE
        self.F = 0.8   # Differential weight for DE
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)
        
        # Evaluate initial particles
        current_scores = np.array([func(p) for p in particles])
        num_evals = self.population_size
        
        # Update personal bests
        better_mask = current_scores < personal_best_scores
        personal_best_scores = np.where(better_mask, current_scores, personal_best_scores)
        personal_best_positions = np.where(better_mask[:, np.newaxis], particles, personal_best_positions)
        
        # Global best initialization
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        # Main optimization loop
        while num_evals < self.budget:
            # Particle Swarm Optimization step
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - particles) +
                          self.c2 * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            # Evaluate the particles
            current_scores = np.array([func(p) for p in particles])
            num_evals += self.population_size

            # Update personal bests
            better_mask = current_scores < personal_best_scores
            personal_best_scores = np.where(better_mask, current_scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], particles, personal_best_positions)

            # Update global best
            current_global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_idx] < global_best_score:
                global_best_position = personal_best_positions[current_global_best_idx]
                global_best_score = personal_best_scores[current_global_best_idx]

            if num_evals >= self.budget:
                break

            # Differential Evolution step
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = particles[a] + self.F * (particles[b] - particles[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, particles[i])
                trial_score = func(trial_vector)
                num_evals += 1
                
                if trial_score < current_scores[i]:
                    particles[i] = trial_vector
                    current_scores[i] = trial_score
                    if trial_score < personal_best_scores[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_scores[i] = trial_score
                        if trial_score < global_best_score:
                            global_best_position = trial_vector
                            global_best_score = trial_score

                if num_evals >= self.budget:
                    break

        return global_best_position, global_best_score