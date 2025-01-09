import numpy as np

class HybridPSOwithADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.F = 0.8
        self.CR = 0.9
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in personal_best_positions])
        
        # Global best position
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_positions - positions) 
                          + self.c2 * r2 * (global_best_position - positions))
            
            # Update positions
            positions = np.clip(positions + velocities, *self.bounds)
            
            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                # Select three random indices not equal to i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = np.clip(positions[a] + self.F * (positions[b] - positions[c]), *self.bounds)
                trial_vector = np.copy(positions[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == self.dim - 1:
                        trial_vector[j] = mutant_vector[j]
                
                # Evaluate trial vector
                trial_score = func(trial_vector)
                eval_count += 1
                
                # Select the better position between current and trial
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    # Update global best if needed
                    if trial_score < personal_best_scores[global_best_idx]:
                        global_best_idx = i
                        global_best_position = trial_vector
        
        return global_best_position

# Example usage:
# optimizer = HybridPSOwithADE(budget=1000, dim=10)
# best_position = optimizer(some_black_box_function)