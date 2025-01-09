import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        self.w = 0.7   # inertia weight
        self.F = 0.8   # DE scaling factor
        self.CR = 0.9  # DE crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def chaotic_init(self, size, dim):
        x = np.random.uniform(0, 1, (size, dim))
        return self.lower_bound + (self.upper_bound - self.lower_bound) * np.abs(np.sin(np.pi * x))

    def __call__(self, func):
        # Initialize the swarm using chaotic map
        positions = self.chaotic_init(self.swarm_size, self.dim)
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        # Evaluate the initial swarm
        scores = np.array([func(pos) for pos in positions])
        num_evaluations = self.swarm_size
        
        # Update personal bests
        improved = scores < personal_best_scores
        personal_best_positions[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]
        
        # Initialize the global best
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        while num_evaluations < self.budget:
            # Adaptive weight
            self.w = 0.5 + 0.4 * (1 - num_evaluations / self.budget)
            r1, r2 = np.random.rand(2)
            velocities = self.w * velocities + \
                         self.c1 * r1 * (personal_best_positions - positions) + \
                         self.c2 * r2 * (global_best_position - positions)
            positions += velocities
            
            # Clip positions to bounds
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            # DE mutation and crossover
            for i in range(self.swarm_size):
                if num_evaluations >= self.budget:
                    break
                idxs = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                a, b, c = positions[idxs]
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.copy(positions[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == self.dim - 1:
                        trial[j] = mutant_vector[j]
                
                trial_score = func(trial)
                num_evaluations += 1
                
                if trial_score < scores[i]:
                    positions[i] = trial
                    scores[i] = trial_score

            # Update personal and global bests
            improved = scores < personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]
            
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]
        
        return global_best_position, personal_best_scores[global_best_index]