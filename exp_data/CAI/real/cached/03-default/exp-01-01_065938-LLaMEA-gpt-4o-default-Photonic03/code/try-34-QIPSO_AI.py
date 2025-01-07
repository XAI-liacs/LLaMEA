import numpy as np

class QIPSO_AI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.9  # Initial inertia weight
        self.w_min = 0.4
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        p_best = pop.copy()
        p_best_fitness = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best = pop[g_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update particle velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (p_best[i] - pop[i]) +
                                 self.c2 * r2 * (g_best - pop[i]))

                # Quantum-inspired position update
                new_position = pop[i] + velocities[i] + np.random.randn(self.dim) * 0.1
                new_position = np.clip(new_position, lb, ub)

                new_fitness = func(new_position)
                evaluations += 1

                # Update personal best and global best
                if new_fitness < p_best_fitness[i]:
                    p_best[i] = new_position
                    p_best_fitness[i] = new_fitness
                    if new_fitness < fitness[g_best_idx]:
                        g_best_idx = i
                        g_best = new_position

                pop[i] = new_position

            # Adapt inertia weight
            self.w = max(self.w_min, self.w - 0.001)

        return g_best