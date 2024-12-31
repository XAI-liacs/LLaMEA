import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.5, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        # Initialize particles
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.Inf)

        global_best_position = None
        global_best_value = np.Inf
        
        evaluations = 0
        
        while evaluations < self.budget:
            inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)  # Dynamic inertia adjustment
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                
                # Evaluate current position
                f_value = func(positions[i])
                evaluations += 1
                
                # Update personal best
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i]
                    
                # Update global best
                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = positions[i]
            
            # Update velocities and positions
            for i in range(self.num_particles):
                # Random coefficients
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                velocities[i] = (inertia_weight * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt