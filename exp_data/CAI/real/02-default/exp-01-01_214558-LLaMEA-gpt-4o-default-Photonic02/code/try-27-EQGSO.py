import numpy as np

class EQGSO:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia=0.5, cognitive=2, social=2, entanglement_prob=0.2, entanglement_strength=0.1):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.entanglement_prob = entanglement_prob
        self.entanglement_strength = entanglement_strength
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [swarm.copy() for swarm in swarms]
        personal_best_values = [np.full(self.swarm_size, float('inf')) for _ in range(self.num_swarms)]

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                for i in range(self.swarm_size):
                    position = swarms[swarm_id][i]
                    value = func(position)
                    self.evaluations += 1

                    if value < personal_best_values[swarm_id][i]:
                        personal_best_values[swarm_id][i] = value
                        personal_best_positions[swarm_id][i] = position

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if np.random.rand() < self.entanglement_prob:
                        position = self.quantum_entanglement(position, personal_best_positions[swarm_id][i], lb, ub)

                    velocities[swarm_id][i] = (self.inertia * velocities[swarm_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (personal_best_positions[swarm_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[swarm_id][i], lb, ub)
                    swarms[swarm_id][i] = position

                    if self.evaluations >= self.budget:
                        break

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def quantum_entanglement(self, position, personal_best_position, lb, ub):
        entangled_position = position + self.entanglement_strength * (personal_best_position - position) * (np.random.rand(self.dim) - 0.5)
        return np.clip(entangled_position, lb, ub)