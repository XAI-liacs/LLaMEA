import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.inf)
        global_best_position = None
        
        evaluation_count = 0
        
        while evaluation_count < self.budget:
            for i in range(self.num_particles):
                # Evaluate
                score = func(particles[i])
                evaluation_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]
                    global_best_position = particles[i]

            if global_best_position is None:
                continue
                
            # Update inertia weight
            inertia_weight = 0.9 - (0.9 - 0.4) * (evaluation_count / self.budget)
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = 2.0 * r1 * (personal_best_positions[i] - particles[i])
                social_component = 2.0 * r2 * (global_best_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Update particle position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                
                # Break if budget is exhausted
                if evaluation_count >= self.budget:
                    break

        return self.f_opt, self.x_opt