import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 10 + int(2.5 * np.log(self.dim))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')
        
        self.f_cr = 0.5
        self.f_f = 0.8
    
    def __call__(self, func):
        for it in range(self.budget):
            for i in range(self.population_size):
                # Evaluate particle
                value = func(self.particles[i])
                
                # Update personal best
                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                # Update global best
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i].copy()

            # Update velocities and positions using PSO
            inertia_weight = 0.5 + 0.4 * np.random.rand()
            cognitive = 2.1 * np.random.rand(self.population_size, self.dim) * (self.personal_best_positions - self.particles)  # Slightly increased
            social = 2.1 * np.random.rand(self.population_size, self.dim) * (self.global_best_position - self.particles)  # Slightly increased
            self.velocities = inertia_weight * self.velocities + cognitive + social
            self.particles += self.velocities
            
            # Boundary handling for particles
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)
            
            # Adaptive Differential Evolution with added perturbation
            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + self.f_f * (b - c) + 0.01 * np.random.randn(self.dim)  # Added perturbation
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                trial_vector = np.copy(self.particles[i])
                cross_points = np.random.rand(self.dim) < self.f_cr
                trial_vector[cross_points] = mutant_vector[cross_points]
                
                # Evaluate trial vector
                trial_value = func(trial_vector)
                
                # Improved Selection
                if trial_value < self.personal_best_values[i] or np.random.rand() < 0.05:  # Stochastic acceptance
                    self.particles[i] = trial_vector
                    self.personal_best_values[i] = trial_value
                    self.personal_best_positions[i] = trial_vector.copy()
                    
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial_vector.copy()
        
        return self.global_best_position, self.global_best_value