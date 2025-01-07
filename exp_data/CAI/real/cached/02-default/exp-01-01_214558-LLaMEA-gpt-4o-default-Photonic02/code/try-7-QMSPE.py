import numpy as np

class QMSPE:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, alpha=0.75, beta=0.25):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        p_best_positions = [np.copy(swarm) for swarm in swarms]
        p_best_values = [np.full(self.swarm_size, float('inf')) for _ in range(self.num_swarms)]

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                for i in range(self.swarm_size):
                    position = self.q_update_position(swarms[swarm_id][i], best_global_position, lb, ub)
                    swarms[swarm_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < p_best_values[swarm_id][i]:
                        p_best_values[swarm_id][i] = value
                        p_best_positions[swarm_id][i] = position

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if self.evaluations >= self.budget:
                        break

                self.evolve_swarm(swarms[swarm_id], p_best_positions[swarm_id], lb, ub)

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def q_update_position(self, position, best_global_position, lb, ub):
        if best_global_position is None:
            return position
        mean = self.alpha * position + (1 - self.alpha) * best_global_position
        levy_step = self.beta * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = mean + levy_step
        return np.clip(new_position, lb, ub)

    def evolve_swarm(self, swarm, p_best_positions, lb, ub):
        for i in range(self.swarm_size):
            if np.random.rand() < 0.1:
                mutation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                swarm[i] = np.clip(swarm[i] + mutation, lb, ub)

        for i in range(0, self.swarm_size, 2):
            if i+1 < self.swarm_size and np.random.rand() < 0.2:
                crossover_point = np.random.randint(1, self.dim)
                swarm[i][:crossover_point], swarm[i+1][:crossover_point] = (
                    swarm[i+1][:crossover_point].copy(), swarm[i][:crossover_point].copy())