import numpy as np
from scipy.optimize import minimize

class PSONMSimplexOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        swarm_size = 15 + 2 * self.dim
        inertia_weight = 0.5
        cognitive_coef = 1.5
        social_coef = 1.5
        
        # Initialize the swarm
        swarm = self.initialize_swarm(swarm_size)
        velocities = np.random.uniform(-1, 1, (swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_fitness = np.array([func(ind) for ind in swarm])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        eval_count = swarm_size

        while eval_count < self.budget:
            for i in range(swarm_size):
                if eval_count >= self.budget:
                    break
                
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coef * r1 * (personal_best_positions[i] - swarm[i]) +
                                 social_coef * r2 * (global_best_position - swarm[i]))

                # Update position
                swarm[i] = np.clip(swarm[i] + velocities[i], self.lb, self.ub)

                # Enforce periodicity in the swarm
                swarm[i] = self.enforce_periodicity(swarm[i])

                # Evaluate fitness
                fitness = func(swarm[i])
                eval_count += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = swarm[i]

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = swarm[i]

            # Local Nelder-Mead Simplex improvement
            if eval_count < self.budget:
                result = minimize(func, global_best_position, bounds=list(zip(self.lb, self.ub)), method='Nelder-Mead')
                eval_count += result.nfev
                if result.fun < global_best_fitness:
                    global_best_position = result.x
                    global_best_fitness = result.fun

        return global_best_position

    def initialize_swarm(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def enforce_periodicity(self, vector):
        period = 2
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector