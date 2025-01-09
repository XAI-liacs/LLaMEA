import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        # Initialize particle positions and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        
        # Initialize personal and global bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        self.iterations = self.budget // self.pop_size
        
    def __call__(self, func):
        evaluations = 0
        w_max = 0.9
        w_min = 0.4
        c1 = c2 = 2.0
        
        for i in range(self.iterations):
            # Adaptive inertia weight
            w = w_max - ((w_max - w_min) * i / self.iterations)
            
            # Evaluate fitness
            scores = np.apply_along_axis(func, 1, self.positions)
            evaluations += self.pop_size
            
            # Update personal and global bests
            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]
            
            global_best_candidate = np.min(scores)
            if global_best_candidate < self.global_best_score:
                self.global_best_score = global_best_candidate
                self.global_best_position = self.positions[np.argmin(scores)]
                
            # Update velocities and positions
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.velocities = (w * self.velocities 
                               + c1 * r1 * (self.personal_best_positions - self.positions) 
                               + c2 * r2 * (self.global_best_position - self.positions))
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
            
            if evaluations >= self.budget:
                break
        
        return self.global_best_position, self.global_best_score