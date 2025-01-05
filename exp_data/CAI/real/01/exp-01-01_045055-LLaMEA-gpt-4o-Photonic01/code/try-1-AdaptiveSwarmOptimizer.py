import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        n_particles = 30
        inertia_weight = 0.7
        cognitive_coeff = 1.5
        social_coeff = 1.5

        # Initialization
        particles = np.random.uniform(lb, ub, (n_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (n_particles, self.dim))
        pbest_positions = np.copy(particles)
        pbest_scores = np.array([func(p) for p in particles])
        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)
        
        evaluations = n_particles
        
        while evaluations < self.budget:
            for i in range(n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                # Dynamic coefficients
                cognitive_coeff = 1.5 + 0.5 * np.sin(evaluations/100)
                social_coeff = 1.5 + 0.5 * np.cos(evaluations/100)
                # Update velocity
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (pbest_positions[i] - particles[i]) +
                                 social_coeff * r2 * (gbest_position - particles[i]))
                # Update position
                particles[i] += velocities[i]
                # Enforce bounds
                particles[i] = np.clip(particles[i], lb, ub)
                
                current_score = func(particles[i])
                evaluations += 1
                if current_score < pbest_scores[i]:
                    pbest_positions[i] = particles[i]
                    pbest_scores[i] = current_score
                    if current_score < gbest_score:
                        gbest_position = particles[i]
                        gbest_score = current_score

                if evaluations >= self.budget:
                    break

        return gbest_position, gbest_score