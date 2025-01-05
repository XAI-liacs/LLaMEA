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
        
        def levy_flight(Lambda):
            sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                     (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
            u = np.random.normal(0, sigma, size=self.dim)
            v = np.random.normal(0, 1, size=self.dim)
            step = u / abs(v)**(1 / Lambda)
            return step
        
        while evaluations < self.budget:
            dynamic_n_particles = max(8, int(n_particles * (1 - 0.5 * np.tanh((evaluations / self.budget) * np.pi))))  # Changed line
            for i in range(dynamic_n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                if pbest_scores[i] < gbest_score:
                    cognitive_coeff = max(1.0, 2.0 - (pbest_scores[i] / gbest_score))
                else:
                    cognitive_coeff = 1.2
                
                social_coeff = 2.0 - cognitive_coeff + 0.4 * np.tanh(i / n_particles)
                
                inertia_weight = 0.9 * (1 - np.tanh((evaluations / self.budget) * np.pi))  # Changed line
                
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (pbest_positions[i] - particles[i]) +
                                 social_coeff * r2 * (gbest_position - particles[i]))
                
                levy_step = levy_flight(1.5) * 0.01  # Introduced line
                velocities[i] += levy_step * (1 - evaluations/self.budget)  # Changed line
                
                velocities[i] = np.clip(velocities[i], -abs(ub-lb) * 0.15, abs(ub-lb) * 0.15)
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