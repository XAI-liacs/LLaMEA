import numpy as np

class DynamicPSO:
    def __init__(self, budget, dim, swarm_size=20, omega=0.5, phi_p=0.5, phi_g=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g

    def __call__(self, func):
        def initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def update_velocity(position, velocity, pbest, gbest):
            inertia = self.omega
            cognitive = self.phi_p * np.random.rand() * (pbest - position)
            social = self.phi_g * np.random.rand() * (gbest - position)
            return inertia * velocity + cognitive + social

        swarm = initialize_swarm()
        pbest = swarm.copy()
        gbest = swarm[np.argmin([func(p) for p in swarm])]
        velocity = np.zeros_like(swarm)

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                velocity[i] = update_velocity(swarm[i], velocity[i], pbest[i], gbest)
                swarm[i] += velocity[i]
                if func(swarm[i]) < func(pbest[i]):
                    pbest[i] = swarm[i]
                    if func(swarm[i]) < func(gbest):
                        gbest = swarm[i]

        return gbest