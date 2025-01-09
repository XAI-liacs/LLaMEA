import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        omega = 0.5
        phi_p = 0.5
        phi_g = 0.9
        F = 0.5
        CR = 0.9
        
        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate the fitness of each particle
                score = func(self.particles[i])
                self.fitness_evaluations += 1
                
                # Update personal and global bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
            
            # Apply Particle Swarm Optimization update
            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)
                cognitive = phi_p * r_p * (self.personal_best_positions[i] - self.particles[i])
                social = phi_g * r_g * (self.global_best_position - self.particles[i])
                self.velocities[i] = omega * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
            
            # Apply Differential Evolution mutation
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break
                candidates = list(range(self.population_size))
                candidates.remove(i)
                x1, x2, x3 = np.random.choice(candidates, 3, replace=False)
                mutant_vector = self.particles[x1] + F * (self.particles[x2] - self.particles[x3])
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.particles[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                trial_score = func(trial_vector)
                self.fitness_evaluations += 1
                
                if trial_score < self.personal_best_scores[i]:
                    self.particles[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector.copy()
                    
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector.copy()
        
        return self.global_best_position, self.global_best_score