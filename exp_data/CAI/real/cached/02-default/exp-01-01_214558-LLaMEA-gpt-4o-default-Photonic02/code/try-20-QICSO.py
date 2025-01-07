import numpy as np

class QICSO:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia=0.5, cognitive=2, social=2, quantum_prob=0.2, interaction_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.interaction_prob = interaction_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        local_best_positions = [swarm.copy() for swarm in swarms]
        local_best_values = [np.full(self.swarm_size, float('inf')) for _ in range(self.num_swarms)]

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                for i in range(self.swarm_size):
                    position = swarms[swarm_id][i]

                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)

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
                        return best_global_position

                self.coevolve_swarms(swarms, swarm_id, lb, ub)

        return best_global_position

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def coevolve_swarms(self, swarms, swarm_id, lb, ub):
        if np.random.rand() < self.interaction_prob:
            partner_swarm_id = np.random.choice([i for i in range(self.num_swarms) if i != swarm_id])
            partner_swarm = swarms[partner_swarm_id]
            for i in range(self.swarm_size):
                if np.random.rand() < 0.5:
                    crossover_point = np.random.randint(1, self.dim)
                    swarms[swarm_id][i][:crossover_point], partner_swarm[i][:crossover_point] = (
                        partner_swarm[i][:crossover_point].copy(), swarms[swarm_id][i][:crossover_point].copy())