import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, n_particles=30):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.lower_bound = -100.0
        self.upper_bound = 100.0

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        velocities = np.zeros((self.n_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.n_particles, np.inf)
        
        # Initialize global best
        global_best_position = None
        global_best_score = np.inf
        
        # Begin optimization
        evaluations = 0
        stagnation_counter = 0  # New line
        while evaluations < self.budget:
            previous_global_best_score = global_best_score  # New line
            for i in range(self.n_particles):
                if evaluations >= self.budget:
                    break
                
                # Evaluate the fitness of the current particle
                fitness = func(positions[i])
                evaluations += 1
                
                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = positions[i]
            
            if global_best_score == previous_global_best_score:  # New line
                stagnation_counter += 1  # New line
            else:  # New line
                stagnation_counter = 0  # New line
            
            # Update inertia weight dynamically
            inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                # Increment learning factor based on the success of the particle
                learning_factor_cognitive = 2.0 - 1.5 * (evaluations / self.budget) + 0.5 * (personal_best_scores[i] < global_best_score)
                learning_factor_social = 2.0 + 1.5 * (evaluations / self.budget)
                velocities[i] = (inertia_weight * velocities[i] +
                                 learning_factor_cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 learning_factor_social * r2 * (global_best_position - positions[i]))
                # Adaptive mutation mechanism based on stagnation
                mutation_probability = min(0.1 + stagnation_counter * 0.05, 0.5)  # New line
                if np.random.rand() < mutation_probability:  # Changed line
                    velocities[i] += np.random.normal(0, 1, self.dim)
                
                # Adaptive velocity scaling to improve convergence
                velocities[i] = np.clip(velocities[i], -20 * (1 - evaluations/self.budget), 20 * (1 - evaluations/self.budget))
                positions[i] += velocities[i]
                
                # Clamp positions to bounds
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
        
        return global_best_score, global_best_position