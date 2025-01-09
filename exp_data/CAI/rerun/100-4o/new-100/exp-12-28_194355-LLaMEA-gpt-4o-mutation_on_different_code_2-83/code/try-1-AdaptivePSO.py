import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.velocity = np.random.uniform(-1, 1, (num_particles, dim))
        self.position = np.random.uniform(-5, 5, (num_particles, dim))
        self.personal_best = np.copy(self.position)
        self.personal_best_value = np.full(num_particles, np.Inf)
        self.global_best = None
        self.global_best_value = np.Inf

    def __call__(self, func):
        for iteration in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                f_value = func(self.position[i])
                if f_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = f_value
                    self.personal_best[i] = self.position[i]
                if f_value < self.global_best_value:
                    self.global_best_value = f_value
                    self.global_best = self.position[i]
                    self.f_opt = f_value
                    self.x_opt = self.position[i]
            
            self._update_particles(func.bounds.lb, func.bounds.ub, iteration)

        return self.f_opt, self.x_opt

    def _update_particles(self, lb, ub, iteration):
        # Adaptive inertia weight
        inertia_weight = 0.9 - iteration / (self.budget // self.num_particles * 1.5)
        cognitive_coef = 1.5
        social_coef = 1.5
        
        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        cognitive_velocity = cognitive_coef * r1 * (self.personal_best - self.position)
        social_velocity = social_coef * r2 * (self.global_best - self.position)
        
        self.velocity = (inertia_weight * self.velocity) + cognitive_velocity + social_velocity

        # Adaptive velocity reduction
        self.velocity = np.clip(self.velocity, -0.1, 0.1)

        self.position += self.velocity

        # Dynamic boundary check
        self.position = np.where(self.position < lb, lb + np.abs(self.velocity), self.position)
        self.position = np.where(self.position > ub, ub - np.abs(self.velocity), self.position)