import numpy as np

class PSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.9
        self.w_min = 0.4
        self.velocity_clamp = 0.5

    def __call__(self, func):
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
        
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        eval_count = 0
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                current_fitness = func(positions[i])
                eval_count += 1
                if current_fitness < personal_best_scores[i]:
                    personal_best_scores[i] = current_fitness
                    personal_best_positions[i] = positions[i]
                if current_fitness < global_best_score:
                    global_best_score = current_fitness
                    global_best_position = positions[i]

            if eval_count >= self.budget:
                break

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)
            positions += velocities
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)
            
            self.w = self.w_min + (0.9 - self.w_min) * (self.budget - eval_count) / self.budget

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt