import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(30, budget//10)
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.7  # Adjusted from 1.5 to 1.7
        self.social_coef = 1.5
        self.temperature = 1000
        self.cooling_rate = 0.95
        self.velocities = np.random.rand(self.num_particles, dim)
        self.particles = np.random.rand(self.num_particles, dim)
        self.personal_best_positions = np.copy(self.particles)
        self.global_best_position = np.zeros(dim)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        self.global_best_score = float('inf')
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.particles = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.social_coef * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                
                # Update particle position
                new_position = self.particles[i] + self.velocities[i]
                new_position = np.clip(new_position, lb, ub)
                
                # Simulated Annealing local search
                if evaluations < self.budget:
                    candidate_position = new_position + np.random.normal(0, 1, self.dim)
                    candidate_position = np.clip(candidate_position, lb, ub)
                    candidate_score = func(candidate_position)
                    evaluations += 1
                    delta_score = candidate_score - score
                    acceptance_probability = np.exp(-delta_score / self.temperature)
                    if candidate_score < score or np.random.rand() < acceptance_probability:
                        self.particles[i] = candidate_position
                        score = candidate_score
                
                # Update temperature
                self.temperature *= self.cooling_rate
                
        return self.global_best_position