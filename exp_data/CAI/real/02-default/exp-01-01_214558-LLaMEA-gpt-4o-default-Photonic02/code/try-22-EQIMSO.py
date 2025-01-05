import numpy as np

class EQIMSO:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia_bounds=(0.4, 0.9), cognitive=2, social=2, initial_quantum_prob=0.2, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia_bounds = inertia_bounds
        self.cognitive = cognitive
        self.social = social
        self.initial_quantum_prob = initial_quantum_prob
        self.evaluations = 0
        self.diversity_threshold = diversity_threshold

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [swarm.copy() for swarm in swarms]
        personal_best_values = [np.full(self.swarm_size, float('inf')) for _ in range(self.num_swarms)]
        inertia_weights = np.full((self.num_swarms, self.swarm_size), self.inertia_bounds[1])

        while self.evaluations < self.budget:
            for swarm_id in range(self.num_swarms):
                diversity = self.calculate_diversity(swarms[swarm_id])
                dynamic_quantum_prob = self.initial_quantum_prob * (1 + (self.diversity_threshold - diversity))
                
                for i in range(self.swarm_size):
                    position = swarms[swarm_id][i]
                    
                    # Apply adaptive inertia weight
                    inertia = self.adaptive_inertia(inertia_weights[swarm_id][i], personal_best_values[swarm_id][i], best_global_value)
                    
                    # Apply dynamic quantum-inspired perturbation
                    if np.random.rand() < dynamic_quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)

                    velocities[swarm_id][i] = (inertia * velocities[swarm_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (personal_best_positions[swarm_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[swarm_id][i], lb, ub)
                    swarms[swarm_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < personal_best_values[swarm_id][i]:
                        personal_best_values[swarm_id][i] = value
                        personal_best_positions[swarm_id][i] = position

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

    def calculate_diversity(self, swarm):
        centroid = np.mean(swarm, axis=0)
        diversity = np.mean(np.linalg.norm(swarm - centroid, axis=1))
        return diversity

    def adaptive_inertia(self, current_inertia, personal_best_value, global_best_value):
        if personal_best_value < global_best_value:
            return max(self.inertia_bounds[0], current_inertia * 0.99)
        else:
            return min(self.inertia_bounds[1], current_inertia * 1.01)