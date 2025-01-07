import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 5 * dim)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.particles = None
        self.velocities = None
        self.best_positions = None
        self.best_global_position = None
        self.best_personal_scores = None
        self.best_global_score = np.inf
        self.eval_count = 0
        self.stall_count = np.zeros(self.population_size)  # Added line

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_positions = self.particles.copy()
        self.best_personal_scores = np.array([np.inf] * self.population_size)
        
        for _ in range(self.budget):
            scores = np.array([func(p) for p in self.particles])
            self.eval_count += self.population_size
            
            improved_mask = scores < self.best_personal_scores
            self.best_personal_scores[improved_mask] = scores[improved_mask]
            self.best_positions[improved_mask] = self.particles[improved_mask]
            self.stall_count[~improved_mask] += 1  # Added line
            self.stall_count[improved_mask] = 0  # Added line
            
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.best_global_score:
                self.best_global_score = scores[min_score_idx]
                self.best_global_position = self.particles[min_score_idx].copy()

            if self.eval_count >= self.budget:
                break
            
            r1 = np.random.uniform(size=(self.population_size, self.dim))
            r2 = np.random.uniform(size=(self.population_size, self.dim))
            self.velocities = (
                self.w * self.velocities +
                self.c1 * (1 + self.eval_count / self.budget) * r1 * (self.best_positions - self.particles) +
                self.c2 * (0.5 + self.eval_count / (2 * self.budget)) * r2 * (self.best_global_position - self.particles)
            )
            self.particles += 0.9 * self.velocities
            
            out_of_bounds_low = self.particles < lb
            out_of_bounds_high = self.particles > ub
            
            self.particles = np.where(out_of_bounds_low, lb, self.particles)
            self.particles = np.where(out_of_bounds_high, ub, self.particles)
            
            range_adjustment = (ub - lb) * np.exp(-0.01 * self.eval_count / self.budget)
            lb = np.maximum(lb, self.particles.min(axis=0) - range_adjustment)
            ub = np.minimum(ub, self.particles.max(axis=0) + range_adjustment)
            
            self.w *= (0.99 + 0.01 * np.cos(2 * np.pi * self.eval_count / self.budget)) 
            if np.random.rand() < 0.05:
                self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
            
            # Burst exploitation if stalling
            burst_exploit = self.stall_count > 10  # Added line
            self.velocities[burst_exploit] = np.random.uniform(-0.5, 0.5, (burst_exploit.sum(), self.dim))  # Added line
            
        return self.best_global_position, self.best_global_score