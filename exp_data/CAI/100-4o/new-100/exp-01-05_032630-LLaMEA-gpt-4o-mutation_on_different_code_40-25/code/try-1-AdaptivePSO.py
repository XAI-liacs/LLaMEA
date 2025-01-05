import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.Inf)
        
        global_best_position = None
        global_best_value = np.Inf

        w_max, w_min = 0.9, 0.4  # dynamic inertia weight range
        c1 = 1.5  # cognitive parameter
        c2 = 1.5  # social parameter
        
        for eval_count in range(self.budget):
            w = w_max - ((w_max - w_min) * (eval_count / self.budget))  # dynamic inertia
            
            for i in range(self.num_particles):
                f_value = func(positions[i])
                
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = positions[i]

                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = positions[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 c2 * r2 * (global_best_position - positions[i]))

                # Variable mutation rate based on convergence
                mutation_rate = 0.1 + 0.4 * (1 - (global_best_value / np.max(personal_best_values + 1e-10)))
                if np.random.rand() < mutation_rate:
                    velocities[i] += np.random.uniform(-0.5, 0.5, self.dim)
                
                # Update position with boundary check
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
            if global_best_value < self.f_opt:
                self.f_opt = global_best_value
                self.x_opt = global_best_position

        return self.f_opt, self.x_opt