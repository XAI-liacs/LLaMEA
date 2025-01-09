import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.w = 0.7  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover probability

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evals = self.pop_size

        while evals < self.budget:
            # PSO update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_positions - positions)
                          + self.c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, self.lb, self.ub)
            
            # Evaluate fitness
            scores = np.array([func(pos) for pos in positions])
            evals += self.pop_size

            # Update personal bests
            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]
            
            # Update global best
            if np.min(personal_best_scores) < global_best_score:
                global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
                global_best_score = np.min(personal_best_scores)

            # Differential Evolution mutation and crossover
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                mutant_vector = personal_best_positions[indices[0]] + self.f * (personal_best_positions[indices[1]] - personal_best_positions[indices[2]])
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, positions[i])
                
                # Selection
                trial_score = func(trial_vector)
                evals += 1
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    positions[i] = trial_vector
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector
                        if trial_score < global_best_score:
                            global_best_position = trial_vector
                            global_best_score = trial_score

        return global_best_position, global_best_score