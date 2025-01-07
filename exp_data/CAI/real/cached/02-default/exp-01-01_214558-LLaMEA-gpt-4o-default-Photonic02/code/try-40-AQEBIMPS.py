import numpy as np

class AQEBIMPS:
    def __init__(self, budget, dim, num_groups=4, group_size=10, inertia=0.4, cognitive=1.5, social=1.5, initial_quantum_prob=0.15, phases=3, quantum_adjustment=0.05):
        self.budget = budget
        self.dim = dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = initial_quantum_prob
        self.phases = phases
        self.quantum_adjustment = quantum_adjustment
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
                    prev_position = position.copy()
                    
                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)
                    
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

                    if value < func(prev_position):
                        self.quantum_prob = min(1.0, self.quantum_prob + self.quantum_adjustment)
                    else:
                        self.quantum_prob = max(0.0, self.quantum_prob - self.quantum_adjustment)

                    if self.evaluations >= self.budget:
                        break

                if phase_counter % self.phases == 0:
                    self.phase_based_adaptation(groups[group_id], lb, ub)

                if self.evaluations >= self.budget:
                    break
            
            phase_counter += 1

        return best_global_position

    def initialize_group(self, lb, ub):
        return np.random.uniform(lb, ub, (self.group_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
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