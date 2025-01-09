import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, c1=2.0, c2=2.0, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def create_particle():
            return np.random.uniform(-5.0, 5.0, self.dim), np.inf, np.zeros(self.dim)

        def update_particle(particle, g_best):
            position, _, velocity = particle
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = self.f * velocity + self.c1 * r1 * (g_best[0] - position) + self.c2 * r2 * (particle[2] - position)
            new_position = np.clip(position + velocity, -5.0, 5.0)
            return new_position, objective_function(new_position), velocity

        swarm = [create_particle() for _ in range(self.swarm_size)]
        g_best = min(swarm, key=lambda x: x[1])

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                new_position, new_fitness, new_velocity = update_particle(swarm[i], g_best)
                if new_fitness < swarm[i][1]:
                    swarm[i] = new_position, new_fitness, new_velocity
                    if new_fitness < g_best[1]:
                        g_best = new_position, new_fitness, new_velocity

            for i in range(self.swarm_size):
                j, k, l = np.random.choice(range(self.swarm_size), 3, replace=False)
                mutant = swarm[j][0] + self.f * (swarm[k][0] - swarm[l][0])
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, swarm[i][0])
                trial_fitness = objective_function(trial)
                if trial_fitness < swarm[i][1]:
                    swarm[i] = trial, trial_fitness, swarm[i][2]
                    if trial_fitness < g_best[1]:
                        g_best = trial, trial_fitness, swarm[i][2]

        return g_best[0]

# Usage example:
budget = 1000
dim = 10
optimizer = HybridPSODE(budget, dim)
result = optimizer(lambda x: np.sum(np.square(x)))
print(result)