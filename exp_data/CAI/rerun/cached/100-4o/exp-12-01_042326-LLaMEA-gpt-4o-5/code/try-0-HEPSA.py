import numpy as np

class HEPSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([float('inf')] * self.population_size)

        # Evaluate initial particles
        for i in range(self.population_size):
            score = func(particles[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_weight * r1 * (personal_best_positions - particles) +
                          self.social_weight * r2 * (global_best_position - particles))
            particles += velocities
            
            # Clamp positions to bounds
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Evaluate new particles
            for i in range(self.population_size):
                score = func(particles[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score