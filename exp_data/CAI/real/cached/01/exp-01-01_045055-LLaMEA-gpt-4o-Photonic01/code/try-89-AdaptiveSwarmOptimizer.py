import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        n_particles = 30
        inertia_weight = 0.9
        cognitive_coeff = 1.5
        social_coeff = 1.5

        particles = np.random.uniform(lb, ub, (n_particles, self.dim)) * 0.8 + 0.1 * (ub - lb)
        velocities = np.random.uniform(-1, 1, (n_particles, self.dim))
        pbest_positions = np.copy(particles)
        pbest_scores = np.array([func(p) for p in particles])
        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)
        
        evaluations = n_particles
        
        while evaluations < self.budget:
            dynamic_n_particles = max(8, n_particles - int((evaluations / self.budget)**2 * n_particles / 2))
            for i in range(dynamic_n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                if pbest_scores[i] < gbest_score:
                    cognitive_coeff = max(1.0, 2.0 - (pbest_scores[i] / gbest_score))
                else:
                    cognitive_coeff = 1.5 * (1 + 0.1 * np.sin(evaluations / self.budget * np.pi))  # Adjusted line
                
                social_coeff = 2.0 - cognitive_coeff + 0.5 * (i / n_particles) * (0.9 + 0.1 * np.cos(evaluations / self.budget * np.pi))  # Adjusted line
                
                inertia_weight = 0.9 * (1 - (evaluations / self.budget)**1.5) - 0.3 * (1 + np.cos((evaluations / self.budget) * np.pi)) / 2
                inertia_weight *= (0.5 + 0.5 * np.cos(evaluations / self.budget * np.pi))
                
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (pbest_positions[i] - particles[i]) +
                                 social_coeff * r2 * (gbest_position - particles[i]) +
                                 0.1 * np.random.normal(0, 1, self.dim) * (1 - evaluations/self.budget) + 
                                 0.05 * np.sin(evaluations) +
                                 0.1 * (0.7 + 0.3 * np.sin(np.pi * evaluations / self.budget)) * np.random.normal(0, 1, self.dim) * (1 - evaluations/self.budget))
                velocity_clamp_factor = 0.15 * (1 - evaluations / self.budget)
                velocities[i] = np.clip(velocities[i], -abs(ub-lb) * velocity_clamp_factor, abs(ub-lb) * velocity_clamp_factor)
                particles[i] += velocities[i]
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