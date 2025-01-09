import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.max_iter = budget // self.num_particles
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.particle_pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.particle_vel = np.zeros((self.num_particles, self.dim))
        self.global_best_pos = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_val = float('inf')
    
    def __call__(self, func):
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                fitness_val = func(self.particle_pos[i])
                if fitness_val < self.global_best_val:
                    self.global_best_val = fitness_val
                    self.global_best_pos = np.copy(self.particle_pos[i])
                
                # Update particle velocity and position using PSO
                inertia_weight = 0.4
                cognitive_weight = 0.8
                social_weight = 0.8
                r1, r2 = np.random.rand(), np.random.rand()
                self.particle_vel[i] = inertia_weight * self.particle_vel[i] + \
                    cognitive_weight * r1 * (self.global_best_pos - self.particle_pos[i]) + \
                    social_weight * r2 * (self.global_best_pos - self.particle_pos[i])
                self.particle_pos[i] = np.clip(self.particle_pos[i] + self.particle_vel[i], self.lower_bound, self.upper_bound)
                
                # Perform Simulated Annealing for local search
                current_pos = self.particle_pos[i]
                current_val = func(current_pos)
                T = 1.0 - t / self.max_iter  # Annealing schedule
                new_pos = current_pos + np.random.normal(0, T, self.dim)
                new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)
                new_val = func(new_pos)
                if new_val < current_val or np.random.rand() < np.exp((current_val - new_val) / T):
                    self.particle_pos[i] = new_pos

        return self.global_best_pos