import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        personal_best_positions = np.copy(swarm)
        personal_best_values = np.array([func(pos) for pos in swarm])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]
        
        evaluations = self.swarm_size

        while evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)  # Adaptive inertia weight
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best_position - swarm[i]))
                # Update position
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = swarm[i]

                # Apply DE-inspired crossover
                if evaluations < self.budget:
                    idxs = np.random.choice(self.swarm_size, 3, replace=False)
                    a, b, c = personal_best_positions[idxs]
                    mutant = a + 0.8 * (b - c)
                    mutant = np.clip(mutant, lb, ub)
                    
                    trial = np.where(np.random.rand(self.dim) < 0.9, mutant, swarm[i])
                    trial_f_value = func(trial)
                    evaluations += 1
                
                    if trial_f_value < f_value:
                        swarm[i] = trial
                        f_value = trial_f_value
                        if f_value < personal_best_values[i]:
                            personal_best_values[i] = f_value
                            personal_best_positions[i] = trial

                # Update global best
                if personal_best_values[i] < global_best_value:
                    global_best_value = personal_best_values[i]
                    global_best_position = personal_best_positions[i]

        self.f_opt = global_best_value
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt