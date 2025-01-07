import numpy as np

class EQIMSO:
    def __init__(self, budget, dim, base_num_swarms=5, base_swarm_size=10, inertia=0.5, cognitive=2, social=2, quantum_prob=0.2, diversity_threshold=0.1, adaptive_rate=0.05):
        self.budget = budget
        self.dim = dim
        self.base_num_swarms = base_num_swarms
        self.base_swarm_size = base_swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.diversity_threshold = diversity_threshold
        self.adaptive_rate = adaptive_rate
        self.num_swarms = base_num_swarms
        self.swarm_size = base_swarm_size

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
                    
                    if np.random.rand() < dynamic_quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)
                    
                    velocities[swarm_id][i] = (self.inertia * velocities[swarm_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (swarms[swarm_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[swarm_id][i], lb, ub)
                    swarms[swarm_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position
                        self.update_adaptive_parameters(swarm_id)

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
    
    def calculate_diversity(self, swarm):
        centroid = np.mean(swarm, axis=0)
        diversity = np.mean(np.linalg.norm(swarm - centroid, axis=1))
        return diversity

    def update_adaptive_parameters(self, swarm_id):
        progress_ratio = self.evaluations / self.budget
        self.num_swarms = max(1, int(self.base_num_swarms * (1 + progress_ratio)))
        self.swarm_size = max(2, int(self.base_swarm_size * (1 - progress_ratio)))

        # Adjust learning rates
        self.inertia = max(0.4, self.inertia - self.adaptive_rate * progress_ratio)
        self.cognitive = min(2.5, self.cognitive + self.adaptive_rate * progress_ratio)
        self.social = min(2.5, self.social + self.adaptive_rate * progress_ratio)