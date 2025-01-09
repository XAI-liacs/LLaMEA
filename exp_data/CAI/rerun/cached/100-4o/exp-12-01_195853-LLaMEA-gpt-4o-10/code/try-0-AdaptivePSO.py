import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.inertia = 0.9  # Inertia weight factor
        self.evals = 0

    def __call__(self, func):
        # Initialize the swarm
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        self.evals = self.swarm_size
        
        while self.evals < self.budget:
            for i in range(self.swarm_size):
                if self.evals >= self.budget:
                    break
                
                # Update velocities
                velocities[i] = (self.inertia * velocities[i] + 
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - positions[i]))

                # Update positions
                positions[i] += velocities[i]
                
                # Ensure within bounds
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                score = func(positions[i])
                self.evals += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Adapt inertia weight
            self.inertia = 0.4 + 0.5 * ((self.budget - self.evals) / self.budget)
        
        return global_best_position, global_best_score