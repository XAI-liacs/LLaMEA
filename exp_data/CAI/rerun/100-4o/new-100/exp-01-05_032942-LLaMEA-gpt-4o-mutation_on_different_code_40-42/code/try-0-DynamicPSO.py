import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w_max = 0.9
        self.w_min = 0.4

    def __call__(self, func):
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_bests = particles.copy()
        personal_best_values = np.full(self.swarm_size, np.Inf)

        for i in range(self.budget):
            w = self.w_max - (self.w_max - self.w_min) * i / self.budget

            for j in range(self.swarm_size):
                fitness = func(particles[j])
                if fitness < personal_best_values[j]:
                    personal_best_values[j] = fitness
                    personal_bests[j] = particles[j].copy()
                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = particles[j].copy()

            for j in range(self.swarm_size):
                inertia = w * velocities[j]
                cognitive = self.c1 * np.random.rand(self.dim) * (personal_bests[j] - particles[j])
                social = self.c2 * np.random.rand(self.dim) * (self.x_opt - particles[j])
                velocities[j] = inertia + cognitive + social

                velocities[j] = np.clip(velocities[j], -1, 1)
                particles[j] = np.clip(particles[j] + velocities[j], self.bounds[0], self.bounds[1])
        
        return self.f_opt, self.x_opt