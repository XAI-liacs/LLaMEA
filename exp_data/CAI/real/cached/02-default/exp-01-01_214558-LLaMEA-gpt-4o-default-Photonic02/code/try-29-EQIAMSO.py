import numpy as np

class EQIAMSO:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia=0.5, cognitive=2, social=2, quantum_prob=0.2, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.diversity_threshold = diversity_threshold
        self.learning_rate_decay = 0.99  # New parameter for adaptive learning

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                diversity = self.calculate_diversity(swarms[swarm_id])
                dynamic_quantum_prob = self.quantum_prob * (1 + (self.diversity_threshold - diversity))
                
                for i in range(self.swarm_size):
                    position = swarms[swarm_id][i]
                    
                    # Apply dynamic quantum-inspired perturbation
                    if np.random.rand() < dynamic_quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)
                    
                    velocities[swarm_id][i] = (self.inertia * velocities[swarm_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (swarms[swarm_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))

                    # Adaptive learning rate adjustment
                    velocities[swarm_id][i] *= self.learning_rate_decay

                    position = np.clip(position + velocities[swarm_id][i], lb, ub)
                    swarms[swarm_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if self.evaluations >= self.budget:
                        break

                # Local Search Exploitation
                self.exploit_local_search(swarms[swarm_id], func, lb, ub)

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def exploit_local_search(self, swarm, func, lb, ub):
        for i in range(self.swarm_size):
            if np.random.rand() < 0.2:
                local_best = swarm[i]
                for _ in range(3):  # Small number of local search steps
                    candidate = local_best + (np.random.rand(self.dim) - 0.5) * 0.05 * (ub - lb)
                    candidate = np.clip(candidate, lb, ub)
                    if func(candidate) < func(local_best):
                        local_best = candidate
                swarm[i] = local_best

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
        return np.clip(q_position, lb, ub)

    def calculate_diversity(self, swarm):
        centroid = np.mean(swarm, axis=0)
        diversity = np.mean(np.linalg.norm(swarm - centroid, axis=1))
        return diversity