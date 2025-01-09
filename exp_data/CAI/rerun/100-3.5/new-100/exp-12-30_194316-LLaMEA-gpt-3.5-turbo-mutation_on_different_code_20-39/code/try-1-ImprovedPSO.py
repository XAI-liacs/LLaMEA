import numpy as np

class ImprovedPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=20, inertia_min=0.4, inertia_max=0.9, cognitive_weight=1.5, social_weight=1.5, local_search_prob_min=0.1, local_search_prob_max=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.local_search_prob_min = local_search_prob_min
        self.local_search_prob_max = local_search_prob_max
        self.f_opt = np.Inf
        self.x_opt = None

    def local_search(self, x, func):
        x_local = x + np.random.uniform(-0.1, 0.1, size=self.dim)
        f_local = func(x_local)
        if f_local < func(x):
            return x_local, f_local
        else:
            return x, func(x)

    def __call__(self, func):
        swarm_pos = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.swarm_size, self.dim))
        swarm_vel = np.zeros((self.swarm_size, self.dim))
        personal_best_pos = swarm_pos.copy()
        personal_best_val = np.array([func(x) for x in swarm_pos])
        
        for i in range(self.budget):
            global_best_idx = np.argmin(personal_best_val)
            global_best_pos = personal_best_pos[global_best_idx]

            inertia = self.inertia_min + (self.inertia_max - self.inertia_min) * (self.budget - i) / self.budget
            local_search_prob = self.local_search_prob_min + (self.local_search_prob_max - self.local_search_prob_min) * (self.budget - i) / self.budget
            
            for j in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                
                swarm_vel[j] = (inertia * swarm_vel[j] +
                                self.cognitive_weight * r1 * (personal_best_pos[j] - swarm_pos[j]) +
                                self.social_weight * r2 * (global_best_pos - swarm_pos[j]))
                
                swarm_pos[j] = np.clip(swarm_pos[j] + swarm_vel[j], func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < local_search_prob:
                    swarm_pos[j], personal_best_val[j] = self.local_search(swarm_pos[j], func)
                
                if personal_best_val[j] < func(personal_best_pos[j]):
                    personal_best_pos[j] = swarm_pos[j]
                    personal_best_val[j] = func(swarm_pos[j])
                    
                if personal_best_val[j] < self.f_opt:
                    self.f_opt = personal_best_val[j]
                    self.x_opt = personal_best_pos[j]
                    
        return self.f_opt, self.x_opt