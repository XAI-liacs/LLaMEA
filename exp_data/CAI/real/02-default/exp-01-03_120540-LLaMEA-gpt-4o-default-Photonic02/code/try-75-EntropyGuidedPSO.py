import numpy as np

class EntropyGuidedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.5
        self.c2 = 2.5
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = None
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def entropy_based_coefficients(self, evaluations):
        entropy = -np.sum(self.personal_best_scores * np.log(self.personal_best_scores + 1e-10))
        relative_entropy = entropy / np.log(self.population_size)
        w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
        c1_adaptive = self.c1 * (1 - relative_entropy)
        c2_adaptive = self.c2 * relative_entropy
        return w, c1_adaptive, c2_adaptive
    
    def update_particle(self, func, idx, w, c1_adaptive, c2_adaptive):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_component = c1_adaptive * r1 * (self.personal_best_positions[idx] - self.positions[idx])
        social_component = c2_adaptive * r2 * (self.best_position - self.positions[idx])
        self.velocities[idx] = w * self.velocities[idx] + cognitive_component + social_component
        self.positions[idx] += self.velocities[idx]
        self.positions[idx] = np.clip(self.positions[idx], func.bounds.lb, func.bounds.ub)
    
    def update_personal_best(self, idx, score):
        if score < self.personal_best_scores[idx]:
            self.personal_best_scores[idx] = score
            self.personal_best_positions[idx] = np.copy(self.positions[idx])
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Evaluate particle fitness
                score = func(self.positions[i])
                evaluations += 1
                
                # Update personal and global bests
                self.update_personal_best(i, score)
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = np.copy(self.positions[i])
            
            # Adaptive coefficients
            w, c1_adaptive, c2_adaptive = self.entropy_based_coefficients(evaluations)
            
            # Update particles
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                self.update_particle(func, i, w, c1_adaptive, c2_adaptive)