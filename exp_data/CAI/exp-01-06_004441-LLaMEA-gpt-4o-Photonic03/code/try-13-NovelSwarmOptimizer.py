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
        # Initialize particles' positions and velocities
        positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.particle_count, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current positions
            for i in range(self.particle_count):
                if evaluations >= self.budget:
                    break
                current_score = func(positions[i])
                evaluations += 1
                
                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i]

            # Update velocities and positions
            for i in range(self.particle_count):
                if evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_const * r2 * (global_best_position - positions[i]))
                velocities[i] *= 0.9  # Adaptive velocity scaling
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

            # Adaptive inertia weight decay
            self.inertia_weight *= 0.99

            # Stochastic perturbation
            if evaluations < self.budget:
                perturbation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                test_position = np.clip(global_best_position + perturbation, lb, ub)
                test_score = func(test_position)
                evaluations += 1
                if test_score < global_best_score:
                    global_best_score = test_score
                    global_best_position = test_position
            
        return global_best_position