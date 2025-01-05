import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        n_particles = 30  # Initial number of particles
        inertia_weight = 0.9
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
            if evaluations % 100 == 0:  # Adjust the number of particles periodically
                n_particles = min(n_particles + 1, 50)
                new_particles = np.random.uniform(lb, ub, (1, self.dim))
                particles = np.vstack((particles, new_particles))
                velocities = np.vstack((velocities, np.random.uniform(-1, 1, (1, self.dim))))
                pbest_positions = np.vstack((pbest_positions, new_particles))
                pbest_scores = np.append(pbest_scores, func(new_particles[0]))
                evaluations += 1
            
            for i in range(n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_coeff = 1.5 + 0.5 * np.sin(evaluations/100)
                social_coeff = 1.5 + 0.5 * np.cos(evaluations/100)
                inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)**2
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (pbest_positions[i] - particles[i]) +
                                 social_coeff * r2 * (gbest_position - particles[i]))
                velocities[i] = np.clip(velocities[i], -abs(ub-lb) * 0.1, abs(ub-lb) * 0.1)
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