import numpy as np

class QPSTEE:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, quantum_prob=0.2, entropy_factor=0.3):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.entropy_factor = entropy_factor
        self.evaluations = 0
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        group_count = max(2, int(self.entropy_factor * self.dim))
        group_size = self.base_group_size
        particles = self.initialize_particles(group_count, group_size, lb, ub)
        velocities = self.initialize_velocities(group_count, group_size)

        while self.evaluations < self.budget:
            entropy = self.calculate_entropy(particles)
            temporal_entropy = self.calculate_temporal_entropy(entropy)

            adaptive_group_count = max(2, int(self.entropy_factor * temporal_entropy * self.dim))

            for group_id in range(adaptive_group_count):
                local_best_position = None
                local_best_value = float('inf')
                
                for i in range(group_size):
                    position = particles[group_id][i]
                    
                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)

                    velocities[group_id][i] = (self.inertia * velocities[group_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (particles[group_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[group_id][i], lb, ub)
                    particles[group_id][i] = position

                    value = func(position)
                    self.evaluations += 1
                    self.history.append(value)

                    if value < local_best_value:
                        local_best_value = value
                        local_best_position = position

                    if self.evaluations >= self.budget:
                        break

                if local_best_value < best_global_value:
                    best_global_value = local_best_value
                    best_global_position = local_best_position

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_particles(self, group_count, group_size, lb, ub):
        return [np.random.uniform(lb, ub, (group_size, self.dim)) for _ in range(group_count)]

    def initialize_velocities(self, group_count, group_size):
        return [np.random.uniform(-1, 1, (group_size, self.dim)) for _ in range(group_count)]

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def calculate_entropy(self, particles):
        flat_positions = np.concatenate(particles).flatten()
        hist, _ = np.histogram(flat_positions, bins='auto', density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def calculate_temporal_entropy(self, current_entropy):
        if not self.history:
            return current_entropy
        diff_entropy = np.abs(np.diff(self.history[-min(len(self.history), 10):]))
        mean_diff = np.mean(diff_entropy) if diff_entropy.size > 0 else 0
        return current_entropy * (1 + mean_diff)