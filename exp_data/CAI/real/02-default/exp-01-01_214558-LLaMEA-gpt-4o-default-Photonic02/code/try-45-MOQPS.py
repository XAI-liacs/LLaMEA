import numpy as np

class MOQPS:
    def __init__(self, budget, dim, num_groups=4, group_size=10, inertia=0.5, cognitive=1.5, social=1.5, quantum_prob=0.2, cooperation_factor=0.6, archive_size=5):
        self.budget = budget
        self.dim = dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.cooperation_factor = cooperation_factor
        self.archive_size = archive_size
        self.evaluations = 0
        self.archive = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        groups = [self.initialize_group(lb, ub) for _ in range(self.num_groups)]
        velocities = [np.random.uniform(-1, 1, (self.group_size, self.dim)) for _ in range(self.num_groups)]
        
        while self.evaluations < self.budget:
            self.update_archive(groups, func)
            
            for group_id in range(self.num_groups):
                local_best_position = None
                local_best_value = float('inf')
                
                for i in range(self.group_size):
                    position = groups[group_id][i]
                    
                    if np.random.rand() < self.quantum_prob:
                        position = self.adaptive_quantum_perturbation(position, lb, ub, len(self.archive))
                    
                    velocities[group_id][i] = (self.inertia * velocities[group_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (groups[group_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[group_id][i], lb, ub)
                    groups[group_id][i] = position

                    value = func(position)
                    self.evaluations += 1

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

            if np.random.rand() < self.cooperation_factor:
                self.cooperative_transfer(groups)

        return best_global_position

    def initialize_group(self, lb, ub):
        return np.random.uniform(lb, ub, (self.group_size, self.dim))

    def adaptive_quantum_perturbation(self, position, lb, ub, archive_length):
        if archive_length == 0:
            scale = 0.1
        else:
            scale = 0.1 * (0.9 ** (archive_length / self.archive_size))
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * scale
        return np.clip(q_position, lb, ub)

    def cooperative_transfer(self, groups):
        for group in groups:
            if np.random.rand() < self.cooperation_factor:
                partner_group = groups[np.random.randint(0, len(groups))]
                partner = partner_group[np.random.randint(0, self.group_size)]
                recipient = group[np.random.randint(0, self.group_size)]
                crossover_point = np.random.randint(1, self.dim)
                recipient[:crossover_point] = partner[:crossover_point].copy()

    def update_archive(self, groups, func):
        for group in groups:
            for individual in group:
                value = func(individual)
                if len(self.archive) < self.archive_size:
                    self.archive.append((individual, value))
                else:
                    self.archive.sort(key=lambda x: x[1])
                    if value < self.archive[-1][1]:
                        self.archive[-1] = (individual, value)