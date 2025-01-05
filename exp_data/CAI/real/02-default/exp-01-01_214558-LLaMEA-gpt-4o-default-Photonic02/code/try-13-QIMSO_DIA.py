import numpy as np

class QIMSO_DIA:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia_min=0.4, inertia_max=0.9, cognitive=2, social=2, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                for i in range(self.swarm_size):
                    position = swarms[swarm_id][i]
                    
                    # Apply quantum-inspired perturbation
                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)
                    
                    local_best_position = min(swarms[swarm_id], key=func)
                    inertia = self.calculate_dynamic_inertia(swarms[swarm_id], lb, ub)

                    velocities[swarm_id][i] = (inertia * velocities[swarm_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (local_best_position - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[swarm_id][i], lb, ub)
                    swarms[swarm_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if self.evaluations >= self.budget:
                        break

                self.evolve_swarm(swarms[swarm_id], lb, ub)

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def evolve_swarm(self, swarm, lb, ub):
        for i in range(self.swarm_size):
            if np.random.rand() < 0.1:
                mutation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                swarm[i] = np.clip(swarm[i] + mutation, lb, ub)

        for i in range(0, self.swarm_size, 2):
            if i+1 < self.swarm_size and np.random.rand() < 0.2:
                crossover_point = np.random.randint(1, self.dim)
                swarm[i][:crossover_point], swarm[i+1][:crossover_point] = (
                    swarm[i+1][:crossover_point].copy(), swarm[i][:crossover_point].copy())

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def calculate_dynamic_inertia(self, swarm, lb, ub):
        diversity = np.mean(np.linalg.norm(swarm - np.mean(swarm, axis=0), axis=1))
        max_diversity = np.linalg.norm(ub - lb) / np.sqrt(self.dim)
        inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * (diversity / max_diversity)
        return inertia