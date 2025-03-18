import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 10
        self.c1_initial = 2.5  # initial cognitive component
        self.c2_initial = 0.5  # initial social component
        self.c1_final = 0.5    # final cognitive component
        self.c2_final = 2.5    # final social component
        self.w_initial = 0.9   # initial inertia weight
        self.w_final = 0.4     # final inertia weight
        self.best_solution = None
        self.best_obj = float('inf')
        
    def initialize_particles(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_obj = np.array([float('inf')] * self.num_particles)
    
    def update_velocities_and_positions(self, evaluations):
        velocity_limit_factor = (self.ub - self.lb) * 0.1  # Adaptive velocity clipping
        progress = evaluations / self.budget
        c1 = self.c1_initial * (1 - progress) + self.c1_final * progress
        c2 = self.c2_initial * (1 - progress) + self.c2_final * progress
        w = self.w_initial * (1 - progress) + self.w_final * progress
        
        for i in range(self.num_particles):
            cognitive_velocity = c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = c2 * np.random.rand(self.dim) * (self.best_solution - self.positions[i])
            self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity
            self.velocities[i] = np.clip(self.velocities[i], -velocity_limit_factor, velocity_limit_factor)
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
    
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
            
            self.update_velocities_and_positions(evaluations)
        
        return self.best_solution