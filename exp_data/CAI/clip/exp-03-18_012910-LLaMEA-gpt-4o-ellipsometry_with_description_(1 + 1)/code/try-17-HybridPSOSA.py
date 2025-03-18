import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(30, int(budget / 50))  # Dynamic population size
        self.inertia_weight = 0.7  # Initial inertia weight
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i, particle in enumerate(particles):
                # Update velocity
                inertia = self.inertia_weight * velocities[i]
                self.cognitive_constant = 1.5 + 0.5 * np.random.rand()  # Adjusted line
                cognitive = self.cognitive_constant * np.random.rand(self.dim) * (personal_best_positions[i] - particle)
                social = self.social_constant * np.random.rand(self.dim) * (global_best_position - particle)
                velocities[i] = inertia + cognitive + social + 0.05 * np.random.uniform(-2, 2, self.dim)  # Adjusted exploration
                # Update position
                particles[i] = np.clip(particle + velocities[i], lb, ub)
                # Evaluate new position
                score = func(particles[i])
                evaluations += 1
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_positions[i], personal_best_scores[i] = particles[i], score
                    if score < global_best_score:  # Strategy change for updating the global best
                        global_best_position, global_best_score = particles[i], score
            # Cool down the temperature
            self.temperature *= self.cooling_rate
            # Update inertia weight adaptively
            self.inertia_weight = max(0.4, self.inertia_weight * 0.99)

        return global_best_position