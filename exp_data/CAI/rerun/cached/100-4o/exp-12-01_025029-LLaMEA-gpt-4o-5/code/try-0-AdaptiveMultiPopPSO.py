import numpy as np

class AdaptiveMultiPopPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.num_pops = 3
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.lb = -5.0
        self.ub = 5.0
        self.best_global_position = None
        self.best_global_value = np.inf

    def initialize_population(self):
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.pop_size, np.inf)
        return positions, velocities, personal_best_positions, personal_best_values

    def __call__(self, func):
        num_evaluations = 0
        populations = [self.initialize_population() for _ in range(self.num_pops)]

        while num_evaluations < self.budget:
            for pop_index, (positions, velocities, personal_best_positions, personal_best_values) in enumerate(populations):
                for i in range(self.pop_size):
                    current_value = func(positions[i])
                    num_evaluations += 1
                    
                    if current_value < personal_best_values[i]:
                        personal_best_positions[i] = positions[i]
                        personal_best_values[i] = current_value

                    if current_value < self.best_global_value:
                        self.best_global_position = positions[i]
                        self.best_global_value = current_value

                    if num_evaluations >= self.budget:
                        break
                
                # Update velocities and positions
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities = (self.w * velocities + 
                              self.c1 * r1 * (personal_best_positions - positions) +
                              self.c2 * r2 * (self.best_global_position - positions))
                positions += velocities
                positions = np.clip(positions, self.lb, self.ub)

                # Adapt strategy based on remaining evaluations
                if num_evaluations / self.budget > 0.7:
                    self.w = 0.3  # Encourage exploitation
                    self.c1, self.c2 = 2.0, 2.0
                else:
                    self.w = 0.5  # Encourage exploration
                    self.c1, self.c2 = 1.5, 1.5

        return self.best_global_position, self.best_global_value