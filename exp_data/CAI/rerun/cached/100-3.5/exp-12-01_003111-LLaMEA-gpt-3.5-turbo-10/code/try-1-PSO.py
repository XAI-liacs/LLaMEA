import numpy as np

class PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.5
        self.c2 = 1.5
        self.global_best_pos = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_val = float('inf')
        self.pop_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.pop_velocities = np.zeros((self.pop_size, self.dim))
        self.local_best_pos = np.copy(self.pop_positions)
        self.local_best_val = np.full(self.pop_size, float('inf'))
        self.inertia_weight = 0.9

    def __call__(self, func):
        evaluate_count = 0
        for iter_count in range(self.max_iter):
            for i in range(self.pop_size):
                fitness_val = func(self.pop_positions[i])
                evaluate_count += 1
                if fitness_val < self.local_best_val[i]:
                    self.local_best_val[i] = fitness_val
                    self.local_best_pos[i] = np.copy(self.pop_positions[i])
                
                if fitness_val < self.global_best_val:
                    self.global_best_val = fitness_val
                    self.global_best_pos = np.copy(self.pop_positions[i])
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.pop_velocities[i] = self.inertia_weight * self.pop_velocities[i] + self.c1 * r1 * (self.local_best_pos[i] - self.pop_positions[i]) + self.c2 * r2 * (self.global_best_pos - self.pop_positions[i])
                self.pop_positions[i] = np.clip(self.pop_positions[i] + self.pop_velocities[i], self.lower_bound, self.upper_bound)
                
                if evaluate_count >= self.budget:
                    return self.global_best_pos
        return self.global_best_pos