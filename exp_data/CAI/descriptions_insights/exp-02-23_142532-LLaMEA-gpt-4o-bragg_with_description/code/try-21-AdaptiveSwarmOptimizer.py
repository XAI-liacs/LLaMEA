import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_comp = 1.5
        self.social_comp = 1.5
        self.learning_rate = 0.5
        self.velocity_clamp = 0.1
    
    def initialize_population(self, bounds):
        """ Initialize particles and velocities within given bounds """
        particles = np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        velocities = np.random.rand(self.population_size, self.dim) * self.velocity_clamp - (self.velocity_clamp / 2)
        return particles, velocities
    
    def update_velocity(self, velocity, particle, personal_best, global_best):
        """ Update velocity based on personal and global best with decay """
        inertia = self.inertia_weight * velocity
        cognitive = self.cognitive_comp * np.random.rand(self.dim) * (personal_best - particle)
        social = self.social_comp * np.random.rand(self.dim) * (global_best - particle)
        decay = 0.99  # Introduced decay factor
        new_velocity = (inertia + cognitive + social) * decay
        return np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp)
    
    def __call__(self, func):
        bounds = func.bounds
        particles, velocities = self.initialize_population(bounds)
        personal_bests = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_idx = np.argmax(personal_best_values)
        global_best = personal_bests[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]
        
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_bests[i], global_best)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds.lb, bounds.ub)
                
                value = func(particles[i])
                evaluations += 1
                
                if value > personal_best_values[i]:
                    personal_bests[i] = particles[i]
                    personal_best_values[i] = value
                
                if value > global_best_value:
                    global_best = particles[i]
                    global_best_value = value

                if evaluations >= self.budget:
                    break

            # Dynamically adjust cognitive and social components based on success
            success_rate = np.mean(personal_best_values > global_best_value)
            self.cognitive_comp += self.learning_rate * (1 - success_rate)
            self.social_comp += self.learning_rate * success_rate
            
        return global_best