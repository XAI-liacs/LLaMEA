import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = min(100, max(10, dim * 2))
        self.position = np.random.uniform(0, 1, (self.particles, self.dim))
        self.velocity = np.zeros((self.particles, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.global_best_position = None
        self.personal_best_value = np.full(self.particles, np.inf)
        self.global_best_value = np.inf
        self.iteration = 0
        self.inertia_weight = 0.9  # Initial inertia weight

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb

        def quantum_update():
            for i in range(self.particles):
                delta = np.random.uniform(-1, 1, self.dim)
                potential_position = self.position[i] + self.velocity[i] * delta
                potential_position = np.clip(potential_position, 0, 1)
                real_position = lb + potential_position * scale
                value = func(real_position)
                if value < self.personal_best_value[i]:
                    self.personal_best_value[i] = value
                    self.personal_best_position[i] = potential_position
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = potential_position

        while self.iteration < self.budget:
            quantum_update()
            for i in range(self.particles):
                cognitive = np.random.random(self.dim)
                social = np.random.random(self.dim)
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +  # Adaptive inertia
                                    cognitive * (self.personal_best_position[i] - self.position[i]) +
                                    social * (self.global_best_position - self.position[i]))
                self.position[i] += 0.9 * self.velocity[i]
                self.position[i] = np.clip(self.position[i], 0, 1)
                # Local search perturbation
                if np.random.random() < 0.1:
                    self.position[i] += np.random.uniform(-0.05, 0.05, self.dim)
                    self.position[i] = np.clip(self.position[i], 0, 1)
            self.inertia_weight *= 0.99  # Decrease inertia weight
            self.iteration += self.particles
            
        return lb + self.global_best_position * scale