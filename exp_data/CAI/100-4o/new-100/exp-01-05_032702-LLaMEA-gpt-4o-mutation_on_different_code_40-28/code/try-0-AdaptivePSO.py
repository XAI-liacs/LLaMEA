import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_max = 0.2 * (5.0 - (-5.0))
        self.v_min = -self.v_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)
        
        # Evaluate initial population
        for i in range(self.swarm_size):
            score = func(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
            if score < self.f_opt:
                self.f_opt = score
                self.x_opt = positions[i]
        
        # Main optimization loop
        evals = self.swarm_size
        while evals < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evals / self.budget)
            for i in range(self.swarm_size):
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (self.x_opt - positions[i]))
                # Clamp velocities
                velocities[i] = np.clip(velocities[i], self.v_min, self.v_max)
                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluate
                score = func(positions[i])
                evals += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = positions[i]
                
                if evals >= self.budget:
                    break
        
        return self.f_opt, self.x_opt