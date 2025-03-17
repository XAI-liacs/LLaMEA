import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.729
        self.cognitive_weight = 1.49445
        self.social_weight = 1.49445
        self.temperature = 100
        self.temp_decay = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        eval_count = self.num_particles

        while eval_count < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_weight * r1 * (personal_best_positions - particles) +
                          self.social_weight * r2 * (global_best_position - particles))
            
            particles = particles + velocities
            particles = np.clip(particles, lb, ub)

            # Evaluate particles and update personal best
            for i in range(self.num_particles):
                score = func(particles[i])
                eval_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                # Simulated Annealing acceptance criterion
                if score < func(global_best_position) or np.exp((func(global_best_position) - score) / self.temperature) > np.random.rand():
                    global_best_position = particles[i]

                if eval_count >= self.budget:
                    break

            # Decay temperature
            self.temperature *= self.temp_decay

            # Adaptive inertia weight adjustment
            self.inertia_weight = 0.9 - (0.5 * (eval_count / self.budget))
        
        return global_best_position