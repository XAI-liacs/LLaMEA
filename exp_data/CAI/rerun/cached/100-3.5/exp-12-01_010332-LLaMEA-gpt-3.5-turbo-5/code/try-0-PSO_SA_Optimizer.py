import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, w=0.5, c1=1.5, c2=1.5, initial_temp=10.0, cooling_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best = swarm.copy()
        global_best = personal_best[np.argmin([cost_function(p) for p in personal_best])]
        temperature = self.initial_temp

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = self.w * velocities[i] + self.c1 * r1 * (personal_best[i] - swarm[i]) + self.c2 * r2 * (global_best - swarm[i])
                new_position = swarm[i] + velocities[i]
                if cost_function(new_position) < cost_function(swarm[i]):
                    swarm[i] = new_position
                    personal_best[i] = new_position
                    if cost_function(new_position) < cost_function(global_best):
                        global_best = new_position
                else:
                    acceptance_probability = np.exp(-(cost_function(new_position) - cost_function(swarm[i])) / temperature)
                    if np.random.rand() < acceptance_probability:
                        swarm[i] = new_position

            temperature *= self.cooling_rate

        return global_best