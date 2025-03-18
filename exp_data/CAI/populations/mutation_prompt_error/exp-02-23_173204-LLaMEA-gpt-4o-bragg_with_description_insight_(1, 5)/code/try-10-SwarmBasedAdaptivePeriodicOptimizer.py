import numpy as np
from scipy.optimize import minimize

class SwarmBasedAdaptivePeriodicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.inertia = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.best_global_position = None
        self.best_global_score = float('inf')

    def initialize_swarm(self, lb, ub):
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.swarm_size, self.dim))
        return positions, velocities

    def periodicity_enforcement(self, position, period=2):
        return np.repeat(np.mean(position.reshape(-1, period), axis=1), period)

    def update_velocity(self, velocity, position, personal_best, global_best):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive = self.cognitive_coefficient * r1 * (personal_best - position)
        social = self.social_coefficient * r2 * (global_best - position)
        return self.inertia * velocity + cognitive + social

    def particle_swarm_optimization(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        positions, velocities = self.initialize_swarm(lb, ub)
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, float('inf'))
        
        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                score = func(positions[i])
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                    
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = positions[i]

            for i in range(self.swarm_size):
                velocities[i] = self.update_velocity(velocities[i], positions[i], personal_best_positions[i], self.best_global_position)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                positions[i] = self.periodicity_enforcement(positions[i])

    def local_optimization(self, func, initial_solution, bounds):
        result = minimize(func, initial_solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        if result.fun < self.best_global_score:
            self.best_global_score = result.fun
            self.best_global_position = result.x

    def __call__(self, func):
        bounds = func.bounds
        self.particle_swarm_optimization(func, bounds)
        
        if self.best_global_position is not None:
            self.local_optimization(func, self.best_global_position, bounds)
        
        return self.best_global_position