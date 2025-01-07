import numpy as np

class AdaptiveMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.population_size = min(50, 5 * dim)
        self.inertia_weight = 0.7
        self.c1 = 2.0
        self.c2 = 2.0
        self.migration_rate = 0.1

    def migrate_particles(self, swarms, global_best):
        for swarm in swarms:
            if np.random.rand() < self.migration_rate:
                idx = np.random.randint(0, len(swarm['positions']))
                swarm['positions'][idx] = global_best

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        swarms = []
        for _ in range(self.num_swarms):
            positions = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            velocities = np.zeros_like(positions)
            personal_best = positions.copy()
            personal_best_values = np.array([func(ind) for ind in positions])
            eval_count = self.population_size
            swarms.append({'positions': positions, 'velocities': velocities, 
                           'personal_best': personal_best, 'personal_best_values': personal_best_values})

        global_best = min(swarms, key=lambda swarm: swarm['personal_best_values'].min())['personal_best'].min(0)
        global_best_value = min(swarm['personal_best_values'].min() for swarm in swarms)

        eval_count = self.num_swarms * self.population_size

        while eval_count < self.budget:
            for swarm in swarms:
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities = (self.inertia_weight * swarm['velocities'][i]
                                  + self.c1 * r1 * (swarm['personal_best'][i] - swarm['positions'][i])
                                  + self.c2 * r2 * (global_best - swarm['positions'][i]))
                    swarm['positions'][i] += velocities
                    swarm['positions'][i] = np.clip(swarm['positions'][i], bounds[:, 0], bounds[:, 1])

                    value = func(swarm['positions'][i])
                    eval_count += 1

                    if value < swarm['personal_best_values'][i]:
                        swarm['personal_best'][i] = swarm['positions'][i]
                        swarm['personal_best_values'][i] = value
                        if value < global_best_value:
                            global_best = swarm['positions'][i]
                            global_best_value = value

                    if eval_count >= self.budget:
                        break

            self.migrate_particles(swarms, global_best)

        return global_best