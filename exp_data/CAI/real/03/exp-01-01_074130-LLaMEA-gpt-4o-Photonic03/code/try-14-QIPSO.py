import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 50
        self.position = np.random.uniform(0, 1, (self.particles, self.dim))
        self.velocity = np.zeros((self.particles, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.global_best_position = None
        self.personal_best_value = np.full(self.particles, np.inf)
        self.global_best_value = np.inf
        self.iteration = 0
        self.chaotic_sequence = np.random.rand(self.particles)  # Chaotic sequence initialization

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        scale = ub - lb

        def logistic_map(x):
            return 4 * x * (1 - x)  # Chaotic logistic map

        def quantum_update():
            for i in range(self.particles):
                delta = np.random.uniform(-1, 1, self.dim) * self.chaotic_sequence[i]
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
            self.chaotic_sequence = logistic_map(self.chaotic_sequence)  # Update chaotic sequence

        while self.iteration < self.budget:
            quantum_update()
            for i in range(self.particles):
                cognitive = np.random.random(self.dim)
                social = np.random.random(self.dim)
                self.velocity[i] = (0.5 * self.velocity[i] +
                                    cognitive * (self.personal_best_position[i] - self.position[i]) +
                                    social * (self.global_best_position - self.position[i]))
                self.position[i] += 0.9 * self.velocity[i]
                self.position[i] = np.clip(self.position[i], 0, 1)
            self.iteration += self.particles

        return lb + self.global_best_position * scale