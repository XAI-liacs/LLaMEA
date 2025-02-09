import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 10
        self.c1 = 2.0  # initial cognitive component
        self.c2 = 2.0  # initial social component
        self.w = 0.9   # initial inertia weight
        self.best_solution = None
        self.best_obj = float('inf')
    
    def initialize_particles(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_obj = np.array([float('inf')] * self.num_particles)
    
    def chaotic_local_search(self, pos):
        chaos_factor = 0.1
        return pos + chaos_factor * np.sin(np.arange(self.dim))
    
    def update_velocities_and_positions(self):
        velocity_limit_factor = (self.ub - self.lb) * 0.1
        for i in range(self.num_particles):
            progress_ratio = self.best_obj / (self.personal_best_obj[i] + np.finfo(float).eps)
            self.c1 = 2.5 - 1.5 * progress_ratio
            self.c2 = 1.5 + 1.5 * progress_ratio
            
            cognitive_velocity = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.c2 * np.random.rand(self.dim) * (self.best_solution - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity
            self.velocities[i] = np.clip(self.velocities[i], -velocity_limit_factor, velocity_limit_factor)
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
            
            # Apply chaotic local search to enhance exploration
            if np.random.rand() < 0.1:
                self.positions[i] = self.chaotic_local_search(self.positions[i])
    
    def __call__(self, func):
        self.initialize_particles(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            objectives = [func(pos) for pos in self.positions]
            evaluations += self.num_particles
            
            for i in range(self.num_particles):
                if objectives[i] < self.personal_best_obj[i]:
                    self.personal_best_obj[i] = objectives[i]
                    self.personal_best_positions[i] = self.positions[i]
                if objectives[i] < self.best_obj:
                    self.best_obj = objectives[i]
                    self.best_solution = self.positions[i]
            
            self.w = 0.9 - 0.6 * (evaluations / self.budget) 
            
            self.update_velocities_and_positions()
        
        return self.best_solution