import numpy as np

class DynamicInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.max_velocity = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_max = 0.9
        self.inertia_min = 0.4

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_position = swarm[np.argmin([objective_function(p) for p in swarm])]
        best_value = objective_function(best_position)

        inertia_weight = self.inertia_max
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = inertia_weight * velocities[i] + self.c1 * r1 * (best_position - swarm[i]) + self.c2 * r2 * (best_position - swarm[i])
                np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)

                current_value = objective_function(swarm[i])
                if current_value < best_value:
                    best_value = current_value
                    best_position = swarm[i]

            inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * _ / self.budget

        return best_value