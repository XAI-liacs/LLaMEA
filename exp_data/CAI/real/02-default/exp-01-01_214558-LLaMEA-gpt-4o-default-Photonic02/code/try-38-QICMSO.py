import numpy as np

class QICMSO:
    def __init__(self, budget, dim, num_groups=4, group_size=10, inertia=0.4, cognitive=1.5, social=1.5, quantum_prob=0.15, phases=3, coop_ratio=0.1):
        self.budget = budget
        self.dim = dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.phases = phases
        self.coop_ratio = coop_ratio
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        groups = [self.initialize_group(lb, ub) for _ in range(self.num_groups)]
        velocities = [np.random.uniform(-1, 1, (self.group_size, self.dim)) for _ in range(self.num_groups)]

        phase_counter = 0

        while self.evaluations < self.budget:
            for group_id in range(self.num_groups):
                for i in range(self.group_size):
                    position = groups[group_id][i]
                    
                    if np.random.rand() < self.quantum_prob:
                        position = self.adaptive_quantum_perturbation(position, lb, ub, phase_counter)
                    
                    velocities[group_id][i] = (self.inertia * velocities[group_id][i] +
                                               self.cognitive * np.random.random(self.dim) * (groups[group_id][i] - position) +
                                               self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                    position = np.clip(position + velocities[group_id][i], lb, ub)
                    groups[group_id][i] = position

                    value = func(position)
                    self.evaluations += 1

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if self.evaluations >= self.budget:
                        break

                if phase_counter % self.phases == 0:
                    self.phase_based_adaptation(groups[group_id], lb, ub, phase_counter)
                    if np.random.rand() < self.coop_ratio:
                        self.cooperative_update(groups, lb, ub)

                if self.evaluations >= self.budget:
                    break
            
            phase_counter += 1

        return best_global_position

    def initialize_group(self, lb, ub):
        return np.random.uniform(lb, ub, (self.group_size, self.dim))

    def adaptive_quantum_perturbation(self, position, lb, ub, phase_counter):
        factor = 0.1 * (1 + phase_counter / (2 * self.phases))
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * factor
        return np.clip(q_position, lb, ub)

    def phase_based_adaptation(self, group, lb, ub, phase_counter):
        mutation_scale = 0.05 * (1 + phase_counter / self.phases)
        for i in range(self.group_size):
            if np.random.rand() < 0.2:
                mutation = (np.random.rand(self.dim) - 0.5) * mutation_scale * (ub - lb)
                group[i] = np.clip(group[i] + mutation, lb, ub)
        
        for i in range(0, self.group_size, 2):
            if i + 1 < self.group_size and np.random.rand() < 0.25:
                crossover_point = np.random.randint(1, self.dim)
                group[i][:crossover_point], group[i + 1][:crossover_point] = (
                    group[i + 1][:crossover_point].copy(), group[i][:crossover_point].copy())

    def cooperative_update(self, groups, lb, ub):
        top_half = int(self.group_size / 2)
        for group in groups:
            best_half = np.argsort([func(position) for position in group])[:top_half]
            for j in range(self.group_size):
                if j not in best_half:
                    donor = groups[np.random.randint(0, self.num_groups)][np.random.randint(0, self.group_size)]
                    group[j] = np.clip(donor + np.random.normal(0, 0.1, self.dim), lb, ub)