import numpy as np

class HybridPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.alpha = 0.01  # Levy flight scaling factor

    def levy_flight(self, scale=1.0):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / np.abs(v) ** (1 / scale)

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([float('inf')] * self.population_size)
        global_best_position = None
        global_best_value = float('inf')

        evaluations = 0
        improvement_rate = 0.1  # Initialize improvement rate

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Evaluate current position
                value = func(particles[i])
                evaluations += 1
                
                # Update personal best
                if value < personal_best_values[i]:
                    improvement_rate = 0.9 * improvement_rate + 0.1  # Increase rate if improvement is made
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i]
                    
                # Update global best
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[i]
            
            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                
                # Apply Adaptive Levy flight for enhanced exploration
                if np.random.rand() < improvement_rate:  # Use improvement_rate to adjust Levy flight application
                    particles[i] += self.alpha * self.levy_flight()

                # Ensure particles stay within bounds
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)
        
        return global_best_position, global_best_value