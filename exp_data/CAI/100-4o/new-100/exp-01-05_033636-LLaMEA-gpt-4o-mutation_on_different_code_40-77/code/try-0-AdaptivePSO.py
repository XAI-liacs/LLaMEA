import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia_weight=(0.9, 0.4), cognitive=2.0, social=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.Inf)
        global_best_position = None
        global_best_score = np.Inf

        count = 0
        
        while count < self.budget:
            for i in range(self.swarm_size):
                if count >= self.budget:
                    break
                
                f = func(particles[i])
                count += 1
                
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = particles[i].copy()
                
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = particles[i].copy()
            
            w = self.inertia_weight[1] + (self.inertia_weight[0] - self.inertia_weight[1]) * ((self.budget - count) / self.budget)

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.social * r2 * (global_best_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
        
        self.f_opt, self.x_opt = global_best_score, global_best_position
        return self.f_opt, self.x_opt