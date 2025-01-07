import numpy as np

class NovelSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = min(30, budget // dim)
        self.inertia_weight = 0.729  # inertia weight
        self.cognitive_const = 1.49445  # cognitive constant
        self.social_const = 1.49445  # social constant

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.particle_count, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.particle_count):
                if evaluations >= self.budget:
                    break
                current_score = func(positions[i])
                evaluations += 1
                
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]
                
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i]

            global_neighbors = np.argsort(personal_best_scores)[:self.particle_count // 2] # Adaptive neighborhood
            local_best_position = np.mean(personal_best_positions[global_neighbors], axis=0) # Memory component

            for i in range(self.particle_count):
                if evaluations >= self.budget:
                    break
                r1, r2, r3 = np.random.rand(3, self.dim) # Adding r3 for memory component
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_const * r2 * (global_best_position - positions[i]) +
                                 0.5 * r3 * (local_best_position - positions[i])) # Memory influence
                velocities[i] *= 0.9
                positions[i] = np.clip(positions[i] + velocities[i]*0.5, lb, ub)

            self.inertia_weight *= 0.99
            progress_ratio = evaluations / self.budget
            self.cognitive_const = 1.5 - 0.5 * progress_ratio
            self.social_const = 1.5 + 0.5 * progress_ratio
            
        return global_best_position