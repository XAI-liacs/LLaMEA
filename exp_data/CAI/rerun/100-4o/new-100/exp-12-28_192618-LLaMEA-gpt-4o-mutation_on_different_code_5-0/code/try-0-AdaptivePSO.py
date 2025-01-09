import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(x) for x in positions])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
                f_val = func(positions[i])
                eval_count += 1
                
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best_positions[i] = positions[i]
                    if f_val < global_best_value:
                        global_best_value = f_val
                        global_best_position = positions[i]

                if eval_count >= self.budget:
                    break
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt