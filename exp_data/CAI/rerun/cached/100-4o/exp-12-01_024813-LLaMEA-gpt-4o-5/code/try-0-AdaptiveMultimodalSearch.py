import numpy as np

class AdaptiveMultimodalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.num_particles = min(50, 10 * dim)  # Define number of particles
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.vel_clamp = 0.5 * (self.bounds[1] - self.bounds[0])
        
    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim)) * self.vel_clamp
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.num_particles):
                score = func(positions[i])
                eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if eval_count >= self.budget:
                    break
            
            if eval_count >= self.budget:
                break

            for i in range(self.num_particles):
                inertia = self.inertia_weight * velocities[i]
                cognitive = self.cognitive_weight * np.random.rand() * (personal_best_positions[i] - positions[i])
                social = self.social_weight * np.random.rand() * (global_best_position - positions[i])

                new_velocity = inertia + cognitive + social
                new_velocity = np.clip(new_velocity, -self.vel_clamp, self.vel_clamp)

                positions[i] += new_velocity
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])
                velocities[i] = new_velocity

        return global_best_position, global_best_score