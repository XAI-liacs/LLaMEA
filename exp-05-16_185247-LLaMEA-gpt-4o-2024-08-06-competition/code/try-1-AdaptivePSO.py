import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        # Initialize particles' positions and velocities
        positions = np.random.uniform(-100, 100, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        
        # Initialize personal and global bests
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        # Evaluation counter
        eval_count = self.num_particles

        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_weight * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_weight * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], -100, 100)

                # Evaluate current position
                current_score = func(positions[i])
                eval_count += 1
                if eval_count >= self.budget:
                    break

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i]

        return global_best_score, global_best_position