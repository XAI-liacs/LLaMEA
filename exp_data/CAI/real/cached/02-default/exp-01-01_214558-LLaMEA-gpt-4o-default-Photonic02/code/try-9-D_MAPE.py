import numpy as np

class D_MAPE:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia=0.5, cognitive=2, social=2):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        # Initialize multiple swarms
        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        local_best_positions = [swarm.copy() for swarm in swarms]
        local_best_values = [np.full(self.swarm_size, float('inf')) for _ in range(self.num_swarms)]

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                for i in range(self.swarm_size):
                    position = swarms[swarm_id][i]
                    velocities[swarm_id][i] = (self.inertia * velocities[swarm_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (local_best_positions[swarm_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[swarm_id][i], lb, ub)
                    swarms[swarm_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < local_best_values[swarm_id][i]:
                        local_best_values[swarm_id][i] = value
                        local_best_positions[swarm_id][i] = position

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if self.evaluations >= self.budget:
                        break

                # Adapt swarm parameters based on performance
                self.adapt_parameters(swarm_id, local_best_values[swarm_id], swarms[swarm_id])

                # Cross-interaction among swarms for diversity
                self.interact_swarms(swarms, velocities, lb, ub)

                # Evolutionary operations (mutation and crossover)
                self.evolve_swarm(swarms[swarm_id], lb, ub)
                
                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def adapt_parameters(self, swarm_id, local_best_values, swarm):
        # Dynamic adjustment based on standard deviation of fitness values
        fitness_std = np.std(local_best_values)
        if fitness_std < 0.1:
            self.inertia = max(0.3, self.inertia - 0.1)
            self.social += 0.1
        else:
            self.inertia = min(0.9, self.inertia + 0.1)
            self.cognitive += 0.1

    def interact_swarms(self, swarms, velocities, lb, ub):
        if np.random.rand() < 0.1:
            for swarm_id in range(self.num_swarms):
                partner_swarm_id = np.random.choice([i for i in range(self.num_swarms) if i != swarm_id])
                partner_idx = np.random.randint(self.swarm_size)
                for i in range(self.swarm_size):
                    interaction = 0.1 * (swarms[partner_swarm_id][partner_idx] - swarms[swarm_id][i])
                    velocities[swarm_id][i] += interaction

    def evolve_swarm(self, swarm, lb, ub):
        # Simple mutation and crossover for diversity
        for i in range(self.swarm_size):
            if np.random.rand() < 0.1:  # Mutation rate
                mutation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                swarm[i] = np.clip(swarm[i] + mutation, lb, ub)
        
        for i in range(0, self.swarm_size, 2):
            if i+1 < self.swarm_size and np.random.rand() < 0.2:  # Crossover rate
                crossover_point = np.random.randint(1, self.dim)
                swarm[i][:crossover_point], swarm[i+1][:crossover_point] = (
                    swarm[i+1][:crossover_point].copy(), swarm[i][:crossover_point].copy())