import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(30, budget // 10)
        self.inertia_weight = 0.7  # Dynamic inertia weight for balanced exploration and exploitation
        self.cognitive_coef = 2.0
        self.social_coef = 2.0
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
            
            phi = self.cognitive_coef + self.social_coef
            chi = 2 / abs(2 - phi - np.sqrt(phi**2 - 4*phi))
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.social_coef * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = chi * (self.velocities[i] +
                                            cognitive_velocity + social_velocity)
                
                # Update particle position with quantum behavior
                new_position = self.particles[i] + self.velocities[i]
                if np.random.rand() < 0.1:  # Quantum tunneling probability
                    new_position = lb + (ub - lb) * np.random.rand(self.dim)
                
                # Dynamic boundary handling
                self.particles[i] = np.where(new_position < lb, lb + 0.1 * (ub - lb), new_position)
                self.particles[i] = np.where(new_position > ub, ub - 0.1 * (ub - lb), new_position)
                
                # Adaptive inertia weight update
                self.inertia_weight = 0.9 - evaluations / self.budget * 0.5
        
        return self.global_best_position