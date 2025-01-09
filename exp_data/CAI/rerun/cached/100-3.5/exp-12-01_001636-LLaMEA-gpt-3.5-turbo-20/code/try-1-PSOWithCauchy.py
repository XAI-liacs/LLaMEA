import numpy as np

class PSOWithCauchy:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.5, cognitive_coef=1.5, social_coef=2.0, cauchy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.cauchy_scale = cauchy_scale

    def __call__(self, func):
        def cauchy_update(particle, gbest):
            return particle + self.cauchy_scale * np.random.standard_cauchy(self.dim)

        swarm = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        pbest = swarm.copy()
        pbest_vals = np.array([func(p) for p in pbest])
        gbest_idx = np.argmin(pbest_vals)
        gbest = pbest[gbest_idx].copy()

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + \
                                self.cognitive_coef * r1 * (pbest[i] - swarm[i]) + \
                                self.social_coef * r2 * (gbest - swarm[i])
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)

                if func(swarm[i]) < func(pbest[i]):
                    pbest[i] = swarm[i].copy()

                    if func(pbest[i]) < func(gbest):
                        gbest = pbest[i].copy()

        return gbest