import numpy as np

class QCMAPE_DGI:
    def __init__(self, budget, dim, num_groups=4, group_size=10, inertia=0.4, cognitive=1.5, social=1.5, quantum_prob=0.15, phases=3, cooperation_factor=0.5, subgroup_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.phases = phases
        self.cooperation_factor = cooperation_factor
        self.subgroup_factor = subgroup_factor
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        groups = [self.initialize_group(lb, ub) for _ in range(self.num_groups)]
        velocities = [np.random.uniform(-1, 1, (self.group_size, self.dim)) for _ in range(self.num_groups)]
        phase_counter = 0

        while self.evaluations < self.budget:
            # Dynamic subgroup interaction
            if phase_counter % (self.phases * 2) == 0:
                self.dynamic_subgroup_transfer(groups)

            for group_id in range(self.num_groups):
                local_best_position = None
                local_best_value = float('inf')
                
                for i in range(self.group_size):
                    position = groups[group_id][i]
                    
                    # Quantum perturbation with adaptive scaling
                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub, phase_counter)
                    
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

                # Update global best if better local found
                if local_best_value < best_global_value:
                    best_global_value = local_best_value
                    best_global_position = local_best_position

                if phase_counter % self.phases == 0:
                    self.phase_based_adaptation(groups[group_id], lb, ub)

                if self.evaluations >= self.budget:
                    break
            
            phase_counter += 1

        return best_global_position

    def initialize_group(self, lb, ub):
        return np.random.uniform(lb, ub, (self.group_size, self.dim))

    def quantum_perturbation(self, position, lb, ub, phase_counter):
        scale = 0.1 * (0.9 ** (phase_counter // self.phases))
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * scale
        return np.clip(q_position, lb, ub)

    def phase_based_adaptation(self, group, lb, ub):
        for i in range(self.group_size):
            if np.random.rand() < 0.2:
                mutation = (np.random.rand(self.dim) - 0.5) * 0.05 * (ub - lb)
                group[i] = np.clip(group[i] + mutation, lb, ub)
        
        for i in range(0, self.group_size, 2):
            if i + 1 < self.group_size and np.random.rand() < 0.25:
                crossover_point = np.random.randint(1, self.dim)
                group[i][:crossover_point], group[i + 1][:crossover_point] = (
                    group[i + 1][:crossover_point].copy(), group[i][:crossover_point].copy())

    def dynamic_subgroup_transfer(self, groups):
        for group in groups:
            if np.random.rand() < self.cooperation_factor:
                subgroup_size = int(self.subgroup_factor * self.group_size)
                subgroup_indices = np.random.choice(self.group_size, size=subgroup_size, replace=False)
                partner_group = groups[np.random.randint(0, len(groups))]
                partner_indices = np.random.choice(self.group_size, size=subgroup_size, replace=False)

                for si, pi in zip(subgroup_indices, partner_indices):
                    recipient = group[si]
                    partner = partner_group[pi]
                    crossover_point = np.random.randint(1, self.dim)
                    recipient[:crossover_point] = partner[:crossover_point].copy()