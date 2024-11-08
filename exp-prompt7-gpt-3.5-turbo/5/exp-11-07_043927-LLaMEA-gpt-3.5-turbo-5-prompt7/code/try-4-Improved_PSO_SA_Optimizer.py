import numpy as np

class Improved_PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_iter=1000, c1=2.0, c2=2.0, initial_temp=10.0, cooling_rate=0.95):
        self.budget, self.dim, self.swarm_size, self.max_iter, self.c1, self.c2, self.initial_temp, self.cooling_rate = budget, dim, swarm_size, max_iter, c1, c2, initial_temp, cooling_rate

    def __call__(self, func):
        obj_func, within_bounds = lambda x: func(x), lambda x: np.clip(x, -5.0, 5.0)
        swarm_pos, swarm_vel = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)), np.zeros((self.swarm_size, self.dim))
        global_best_pos, global_best_val, temperature = np.random.uniform(-5.0, 5.0, self.dim), np.inf, self.initial_temp

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                swarm_vel[i] = 0.3 * swarm_vel[i] + self.c1 * r1 * (global_best_pos - swarm_pos[i]) + self.c2 * r2 * (global_best_pos - swarm_pos[i])
                swarm_pos[i] = within_bounds(swarm_pos[i] + swarm_vel[i])
                fitness_val = obj_func(swarm_pos[i])

                if fitness_val < global_best_val:
                    global_best_val, global_best_pos = fitness_val, np.copy(swarm_pos[i])

                if np.random.rand() < np.exp((global_best_val - fitness_val) / temperature):
                    swarm_pos[i] = within_bounds(swarm_pos[i])

            temperature *= self.cooling_rate

        return global_best_pos