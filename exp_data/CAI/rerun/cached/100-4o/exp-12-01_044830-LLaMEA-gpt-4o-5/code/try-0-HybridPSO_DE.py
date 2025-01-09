import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.5   # inertia weight
        self.F = 0.8   # differential weight for DE
        self.CR = 0.9  # crossover probability for DE
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        # Initialize swarm
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate the current position
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

                # Stop if budget is exhausted
                if evaluations >= self.budget:
                    break
            
            # Update velocities and positions (PSO component)
            r1, r2 = np.random.rand(2)
            for i in range(self.swarm_size):
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                
                # Apply boundary constraints
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])
            
            # Differential Evolution mutation (DE component)
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                indices = np.random.choice(self.swarm_size, 3, replace=False)
                x1, x2, x3 = positions[indices[0]], positions[indices[1]], positions[indices[2]]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                trial = np.copy(positions[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]
                
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial.copy()
        
        return global_best_position, global_best_score