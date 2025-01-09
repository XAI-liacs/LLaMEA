import numpy as np

class HybridPSOSA:
    def __init__(self, budget=10000, dim=10, swarm_size=30, alpha=0.9, temperature=100, inertia_weight=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.temperature = temperature
        self.inertia_weight = inertia_weight
        self.f_opt = np.Inf
        self.x_opt = None

    def levy_flight(self, position, lb, ub):
        levy_step = np.random.standard_cauchy(self.dim)
        position += 0.01 * levy_step
        return np.clip(position, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        for i, position in enumerate(swarm):
            f = func(position)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = position
            if f < personal_best_scores[i]:
                personal_best_scores[i] = f
                personal_best_positions[i] = position

        global_best_position = self.x_opt
        
        eval_count = self.swarm_size
        
        while eval_count < self.budget:
            for i, position in enumerate(swarm):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = r1 * (personal_best_positions[i] - position)
                social = r2 * (global_best_position - position)
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive + social)
                
                swarm[i] = position + velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
                
                if np.random.rand() < 0.3:  # Levy flight based local search with 30% chance
                    swarm[i] = self.levy_flight(swarm[i], lb, ub)
                
                f = func(swarm[i])
                eval_count += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = swarm[i]
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = swarm[i]

            self.inertia_weight *= 0.99  # Decay inertia weight
            self.temperature *= self.alpha
            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        return self.f_opt, self.x_opt