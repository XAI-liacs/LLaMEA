import numpy as np

class EnhancedAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9  
        self.cognitive_comp = 1.5
        self.social_comp = 1.6  
        self.learning_rate = 0.65  
        self.velocity_clamp = 0.1
        self.min_inertia = 0.4  
        self.max_velocity_clamp = 0.2
        self.dynamic_inertia_factor = 0.8
        self.niche_radius = 0.05
        self.adaptive_niche_factor = 1.01

    def initialize_population(self, bounds):
        particles = np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        velocities = np.random.rand(self.population_size, self.dim) * self.velocity_clamp - (self.velocity_clamp / 2)
        return particles, velocities
    
    def update_velocity(self, velocity, particle, personal_best, global_best, inertia_weight):
        inertia = inertia_weight * velocity
        cognitive = self.cognitive_comp * np.random.rand(self.dim) * (personal_best - particle)
        social = self.social_comp * np.random.rand(self.dim) * (global_best - particle)
        new_velocity = inertia + cognitive + social
        adaptive_scaling = 1.05 if np.random.rand() < 0.5 else 0.95  # Adaptive scaling factor
        return np.clip(new_velocity * adaptive_scaling, -self.velocity_clamp * 1.05, self.velocity_clamp * 1.05)
    
    def maintain_diversity(self, particles):
        unique_particles = []
        for i, particle in enumerate(particles):
            if all(np.linalg.norm(particle - p) >= self.niche_radius for p in unique_particles):
                unique_particles.append(particle)
        self.niche_radius *= self.adaptive_niche_factor
        return np.array(unique_particles)

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
            particles = self.maintain_diversity(particles)
            
            for i in range(len(particles)):
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_bests[i], global_best, self.inertia_weight)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds.lb, bounds.ub)
                
                if np.random.rand() < 0.15:  # Increased stochastic perturbation probability
                    particles[i] += np.random.normal(0, 0.01, self.dim)  # Apply small Gaussian noise
                
                value = func(particles[i])
                evaluations += 1
                
                if value > personal_best_values[i]:
                    personal_bests[i] = particles[i]
                    personal_best_values[i] = value
                
                if value > global_best_value:
                    global_best = particles[i]
                    global_best_value = value
                    self.velocity_clamp = min(self.max_velocity_clamp, self.velocity_clamp * 1.08)  # Increased adjustment factor
                    self.learning_rate *= 1.02

                if evaluations >= self.budget:
                    break
            
            self.inertia_weight = max(self.min_inertia, self.inertia_weight * self.dynamic_inertia_factor)
            
            self.cognitive_comp = max(1.0, self.cognitive_comp * (1 + self.learning_rate * 0.1 * np.exp(-evaluations/self.budget)))
            self.social_comp = max(1.0, self.social_comp * (1 + self.learning_rate * 0.1 * np.exp(-evaluations/self.budget)))
            
            self.velocity_clamp = min(self.max_velocity_clamp, self.velocity_clamp * 1.01)
            self.niche_radius = max(0.01, self.niche_radius * 0.98)  # More aggressive adjustment to explore-exploit balance

        return global_best