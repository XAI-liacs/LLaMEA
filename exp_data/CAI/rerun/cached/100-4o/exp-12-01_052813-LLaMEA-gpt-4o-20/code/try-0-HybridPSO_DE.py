import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.current_eval = 0
    
    def __call__(self, func):
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        F = 0.5  # DE scaling factor
        CR = 0.9  # DE crossover rate
        
        while self.current_eval < self.budget:
            # Evaluate the particles
            for i in range(self.population_size):
                if self.current_eval >= self.budget:
                    break
                score = func(self.particles[i])
                self.current_eval += 1
                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
            
            # Update velocities and positions using PSO
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                # Clip to bounds
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
            
            # Apply DE mutation and crossover
            for i in range(self.population_size):
                if self.current_eval >= self.budget:
                    break
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = np.clip(self.particles[a] + F * (self.particles[b] - self.particles[c]), self.lower_bound, self.upper_bound)
                
                # Crossover
                trial_vector = np.copy(self.particles[i])
                crossover_points = np.random.rand(self.dim) < CR
                trial_vector[crossover_points] = mutant_vector[crossover_points]
                
                # Selection
                trial_score = func(trial_vector)
                self.current_eval += 1
                if trial_score < self.personal_best_scores[i]:
                    self.particles[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector
        
        return self.global_best_position