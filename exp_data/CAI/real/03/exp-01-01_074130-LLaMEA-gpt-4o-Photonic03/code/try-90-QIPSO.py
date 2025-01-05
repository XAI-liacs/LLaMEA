import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = min(100, max(10, dim * 2))  # Dynamic particle count
        self.position = np.random.uniform(0, 1, (self.particles, self.dim))
        self.velocity = np.zeros((self.particles, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.global_best_position = None
        self.personal_best_value = np.full(self.particles, np.inf)
        self.global_best_value = np.inf
        self.iteration = 0

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb

        def quantum_update():
            for i in range(self.particles):
                delta = np.random.uniform(-0.5, 0.5, self.dim)  # Enhanced local exploration
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

        inertia_weight = 0.9  # Adaptive inertia weight
        while self.iteration < self.budget:
            quantum_update()
            for i in range(self.particles):
                cognitive = np.random.random(self.dim)
                social = np.random.random(self.dim)
                random_factor = np.random.uniform(0.8, 1.2, self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive * (self.personal_best_position[i] - self.position[i]) +
                                    social * (self.global_best_position - self.position[i])) * random_factor
                self.position[i] += self.velocity[i]  # Removed enhanced velocity scaling
                self.position[i] = np.clip(self.position[i], 0, 1)
            self.iteration += self.particles
            inertia_weight -= 0.5 / self.budget  # Decreasing inertia weight

        return lb + self.global_best_position * scale