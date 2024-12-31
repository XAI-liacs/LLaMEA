import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, pop_size=30, w=0.7, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.pop_size, np.Inf)
        global_best_position = None
        
        function_evaluations = 0

        while function_evaluations < self.budget:
            for idx, particle in enumerate(swarm):
                score = func(particle)
                function_evaluations += 1
                
                if score < personal_best_scores[idx]:
                    personal_best_scores[idx] = score
                    personal_best_positions[idx] = particle
                    
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particle
                    global_best_position = particle

                if function_evaluations >= self.budget:
                    break

            for idx, particle in enumerate(swarm):
                # adaptive inertia weight
                inertia = 0.5 + np.random.rand() * 0.5
                
                # calculate velocity
                r1, r2 = np.random.rand(2, self.dim)
                cognitive = self.c1 * r1 * (personal_best_positions[idx] - particle)
                social = self.c2 * r2 * (global_best_position - particle)
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                
                # update position
                swarm[idx] += velocities[idx]
                swarm[idx] = np.clip(swarm[idx], lb, ub)

        return self.f_opt, self.x_opt