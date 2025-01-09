import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.Inf)
        
        global_best_position = None
        global_best_score = np.Inf
        
        w_max, w_min = 0.9, 0.4
        c1, c2 = 2.5, 1.5
        
        for i in range(self.budget):
            w = w_max - ((w_max - w_min) * (i / self.budget))
            
            for j, particle in enumerate(particles):
                f = func(particle)
                
                if f < personal_best_scores[j]:
                    personal_best_scores[j] = f
                    personal_best_positions[j] = particle
                
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = particle
            
            velocities = (w * velocities 
                          + c1 * np.random.rand(self.population_size, self.dim) * (personal_best_positions - particles)
                          + c2 * np.random.rand(self.population_size, self.dim) * (global_best_position - particles))
            
            particles = np.clip(particles + velocities, lb, ub)
        
        self.f_opt, self.x_opt = global_best_score, global_best_position
        return self.f_opt, self.x_opt